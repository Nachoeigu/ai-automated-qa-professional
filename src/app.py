from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import app as ai_agent
from langchain_groq.chat_models import ChatGroq

app = FastAPI()

class DataInput(BaseModel):
    question: str
    role: str


@app.post("/ask")
async def ask_question(info_input: DataInput):
    model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)

    output = ai_agent.invoke(
        input = {'question':info_input.question, "role": info_input.role}, 
        config = {"configurable":{"classifier_model": model,
                                "section_extractor_model": model,
                                "qa_model": model
                                }
                }
        )
    
    return output['answer']

