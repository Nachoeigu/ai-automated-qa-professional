from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import app as ai_agent
from langchain_groq.chat_models import ChatGroq

app = FastAPI()
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class DataInput(BaseModel):
    question: str
    role: str

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
