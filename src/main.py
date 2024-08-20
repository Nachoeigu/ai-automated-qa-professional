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
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from src.utils import convert_markdown_to_json_if_not_exist
from src.model import QABot, QCBot
from langchain_aws import ChatBedrock


if __name__ == '__main__':
    convert_markdown_to_json_if_not_exist(md_file_name = 'info.md',
                             json_file_name = 'info.json')

    qc_model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature = 0)
    #qa_model = ChatGroq(model = 'llama3-groq-70b-8192-tool-use-preview', temperature = 0)
    #qa_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-exp-0801', temperature = 0)
    #qa_model = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs=dict(temperature=0))
    qa_model = ChatOpenAI(model = 'gpt-4o', temperature = 0)
    
    qc_bot = QCBot(model = qc_model)
    qa_bot = QABot(model = qa_model, qc_bot_chain = qc_bot.chain)
    questions = [
        #'What is your english level? 0) Basic 1) Intermediate 2) Advanced',
        #'Are you open to hybrid job?',
        #"Do you consider a change if we offer 3000 USD per month?",
        #"Cuantos años de experiencia como cientifico de datos usted tiene?",
        #"Did you work with Software Design?",
        #"¿Cuántos años de experiencia tienes con Python?",
        #"How many years of work experience do you have with Python (Programming Language)?",
        "What's your Salary Expectation? (monthly/USD)",
        "What is your level of proficiency in English?\nPossible options: 0) Select an option \n 1) None \n 2) Conversational \n 3) Professional \n 4) Native or bilingual\n Retrieve the number of the correct answer",
    ]
    for question in questions:
        output = qa_bot.chain.invoke({'role': 'AI Developer',
                'question': question
        })
        print(output)
    