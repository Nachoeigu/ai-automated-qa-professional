import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from src.utils import convert_markdown_to_json
from src.model import QABot, QCBot

if __name__ == '__main__':
    convert_markdown_to_json(md_file_name = 'info.md',
                             json_file_name = 'info.json')

    qc_model = ChatGroq(model="llama3-groq-8b-8192-tool-use-preview", temperature = 0)
    qa_model = ChatGroq(model = 'llama3-70b-8192', temperature = 0)
    qc_bot = QCBot(model = qc_model)
    qa_bot = QABot(model = qa_model, qc_bot_chain = qc_bot.chain)
    questions = [
        #'What is your english level? 0) Basic 1) Intermediate 2) Advanced',
        #'Are you open to hybrid job?',
        #"Do you consider a change if we offer 3000 USD per month?",
        #"Cuantos años de experiencia como cientifico de datos usted tiene?",
        #"Did you work with Software Design?",
        "¿Cuántos años de experiencia tienes con Python?"
    ]
    for question in questions:
        output = qa_bot.chain.invoke({'role': 'AI Developer',
                'question': question
        })
        print(output)
    