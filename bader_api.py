from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bader_chatbot import *

app = FastAPI(title="Bader - Nawader Coffee Chatbot API")

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        reply = get_bader_response(request.message, request.user_id)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))