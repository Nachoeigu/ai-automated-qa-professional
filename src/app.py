from fastapi import FastAPI
from pydantic import BaseModel
from src.model import QABot, QCBot
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

app = FastAPI()

# Initialize your models here
qc_model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
qa_model = ChatOpenAI(model='gpt-4o', temperature=0)

qc_bot = QCBot(model=qc_model)
qa_bot = QABot(model=qa_model, qc_bot_chain=qc_bot.chain)

class Question(BaseModel):
    role: str
    question: str

@app.post("/ask")
async def ask_question(question: Question):
    output = qa_bot.chain.invoke({
        'role': question.role,
        'question': question.question
    })
    return {"response": output['reply']}
