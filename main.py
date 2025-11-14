import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from database import db, create_document, get_documents
from schemas import MemoryNote, ConversationTurn
import requests

app = FastAPI(title="Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Study Assistant Backend Running"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

# Simple language detection using LibreTranslate (no key required) or fallback
LIBRE_URL = os.getenv("LIBRE_URL", "https://libretranslate.de/detect")
TRANSLATE_URL = os.getenv("TRANSLATE_URL", "https://libretranslate.de/translate")

class TextIn(BaseModel):
    text: str
    target_lang: Optional[str] = None

@app.post("/api/detect")
def detect_language(payload: TextIn):
    try:
        r = requests.post(LIBRE_URL, data={"q": payload.text})
        data = r.json()
        if isinstance(data, list) and data:
            return {"language": data[0].get("language", "auto"), "confidence": data[0].get("confidence", 0)}
    except Exception:
        pass
    return {"language": "auto", "confidence": 0}

@app.post("/api/translate")
def translate_text(payload: TextIn):
    target = payload.target_lang or "en"
    try:
        r = requests.post(TRANSLATE_URL, data={"q": payload.text, "source": "auto", "target": target})
        data = r.json()
        if isinstance(data, dict) and "translatedText" in data:
            return {"translated": data["translatedText"], "target": target}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"translated": payload.text, "target": target}

# Memorize notes
@app.post("/api/memory")
def save_memory(note: MemoryNote):
    try:
        inserted_id = create_document("memorynote", note)
        return {"id": inserted_id, "status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory")
def list_memory(tag: Optional[str] = None, limit: int = 50):
    try:
        filter_dict = {"tags": {"$in": [tag]}} if tag else {}
        docs = get_documents("memorynote", filter_dict, limit)
        # Convert ObjectId to string
        for d in docs:
            if "_id" in d:
                d["id"] = str(d.pop("_id"))
        return {"items": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Conversation history
@app.post("/api/conversation")
def log_conversation(turn: ConversationTurn):
    try:
        inserted_id = create_document("conversationturn", turn)
        return {"id": inserted_id, "status": "logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation")
def get_conversation(session_id: Optional[str] = None, limit: int = 100):
    try:
        filter_dict = {"session_id": session_id} if session_id else {}
        docs = get_documents("conversationturn", filter_dict, limit)
        for d in docs:
            if "_id" in d:
                d["id"] = str(d.pop("_id"))
        return {"items": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization using external free service (Simple)
SUMMARIZE_URL = os.getenv("SUMMARIZE_URL", "https://r.jina.ai/http://example.com")

class SummarizeIn(BaseModel):
    text: str
    target_lang: Optional[str] = None

@app.post("/api/summarize")
def summarize(payload: SummarizeIn):
    # We'll use a simple heuristic summary: call a remote summarizer if available; otherwise basic extractive
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # naive extractive summary: first 3 sentences fallback
    summary = None
    try:
        # Jina Reader trick: fetches and returns content for a URL. Not helpful for raw text, so skip.
        pass
    except Exception:
        pass

    # Fallback summary by sentence splitting
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 3:
        summary = text
    else:
        # pick first, middle, last sentences as a crude summary
        mid = len(sentences) // 2
        chosen = [sentences[0], sentences[mid], sentences[-1]]
        summary = " ".join(chosen)

    # Translate summary if target_lang specified
    if payload.target_lang:
        try:
            r = requests.post(TRANSLATE_URL, data={"q": summary, "source": "auto", "target": payload.target_lang})
            data = r.json()
            if isinstance(data, dict) and "translatedText" in data:
                summary = data["translatedText"]
        except Exception:
            pass

    return {"summary": summary}

# Q&A using retrieval from saved memory (keyword match) + simple heuristic answer
class QuestionIn(BaseModel):
    question: str
    target_lang: Optional[str] = None

@app.post("/api/ask")
def ask_question(payload: QuestionIn):
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    # rudimentary keyword retrieval from memory
    keywords = [w.lower() for w in q.split() if len(w) > 3]
    try:
        # search notes containing any keyword
        filter_dict = {"$or": [{"content": {"$regex": k, "$options": "i"}} for k in keywords]} if keywords else {}
        mems = get_documents("memorynote", filter_dict, 20)
    except Exception:
        mems = []

    # create a simple answer using found notes
    context_snippets = []
    for m in mems:
        context_snippets.append(m.get("content", ""))
    context = "\n".join(context_snippets)[:2000]

    if context:
        answer = f"Based on your saved notes, here's what seems relevant:\n{context}\n\nIn summary: "
        # reuse summarizer to craft a short answer
        try:
            import re
            sentences = re.split(r"(?<=[.!?])\s+", context)
            if len(sentences) > 2:
                mid = len(sentences) // 2
                answer += " ".join([sentences[0], sentences[mid], sentences[-1]])
            else:
                answer += context
        except Exception:
            answer += context
    else:
        answer = "I couldn't find related notes yet. Try saving key facts first, or ask a more specific question."

    # translate if requested
    if payload.target_lang:
        try:
            r = requests.post(TRANSLATE_URL, data={"q": answer, "source": "auto", "target": payload.target_lang})
            data = r.json()
            if isinstance(data, dict) and "translatedText" in data:
                answer = data["translatedText"]
        except Exception:
            pass

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
