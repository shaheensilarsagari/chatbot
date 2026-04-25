from pathlib import Path
import re

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import wikipedia
from transformers import pipeline
import uvicorn

# ============================================================
# Path setup
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Create FastAPI app
# ============================================================
app = FastAPI()

# Serve static files and templates safely
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static") 
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ============================================================
# Load summarization model
# Use a lighter model for easier student usage
# ============================================================
print("Loading Hugging Face model...")

generator = pipeline("text-generation", model="gpt2")

print("Model loaded successfully.")

# ============================================================
# Helper functions
# ============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_wikipedia(query: str):
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return None, None

        page = wikipedia.page(search_results[0], auto_suggest=False)
        content = clean_text(page.content[:1500])
        return content, page.url

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            content = clean_text(page.content[:1500])
            return content, page.url
        except Exception:
            return None, None

    except wikipedia.exceptions.PageError:
        return None, None

    except Exception:
        return None, None


def summarize_with_huggingface(text: str) -> str:
    try:
        result = generator(
            text,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        return clean_text(result[0]["generated_text"])
    except Exception as e:
        return f"Could not generate summary. Error: {str(e)}"


def generate_simple_answer(question: str) -> str:
    q = question.lower()

    if any(word in q for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "Hello! I am your study assistant. Ask me any topic and I will try to help you."

    if "how are you" in q:
        return "I am fine and ready to help you learn."

    if any(phrase in q for phrase in ["who are you", "what are you", "your name"]):
        return "I am StudyBot, a simple AI chatbot made for students."

    if any(word in q for word in ["thank", "thanks", "thank you"]):
        return "You're welcome. Ask me another question anytime."

    return (
        "I could not find exact information for that question. "
        "Please ask a clearer topic-based question like 'What is gravity?' or 'Explain machine learning'."
    )

# ============================================================
# Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request,"index.html")


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()

        if not question:
            return JSONResponse({
                "answer": "Please type a question first.",
                "source": "System",
                "source_url": ""
            })

        wiki_content, wiki_url = search_wikipedia(question)

        if wiki_content:
            summary = summarize_with_huggingface(wiki_content)
            return JSONResponse({
                "answer": summary,
                "source": "Wikipedia + Hugging Face Summary",
                "source_url": wiki_url
            })

        fallback_answer = generate_simple_answer(question)
        return JSONResponse({
            "answer": fallback_answer,
            "source": "StudyBot Response",
            "source_url": ""
        })

    except Exception as e:
        return JSONResponse(
            {
                "answer": f"Backend error: {str(e)}",
                "source": "Server Error",
                "source_url": ""
            },
            status_code=500
        )

# ============================================================
# Run app
# ============================================================
if __name__ == "__main__":
    print("Starting StudyBot server...")
    print("Open in browser: http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000)