from fastapi import FastAPI, HTTPException, Request
import os
from pydantic import BaseModel
from src.agent import app as ai_agent
from langchain_groq.chat_models import ChatGroq
import logging

logger = logging.getLogger()
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class DataInput(BaseModel):
    question: str
    role: str
    
app = FastAPI()

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = request.headers.get("apikey")
    if api_key != os.getenv("FASTAPI_API_KEY"):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")
    response = await call_next(request)
    return response



@app.post("/ask")
async def ask_question(info_input: DataInput):
    output = ai_agent.invoke(
            input = {'question':info_input.question, "role": info_input.role}, 
            config = {"configurable":{"classifier_model": model,
                                        "section_extractor_model": model,
                                        "qa_model": model,
                                        }
                        }
            )


    return output['answer']
