import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from src.utils import convert_markdown_to_json
from src.model import QABot, QCBot

if __name__ == '__main__':
    convert_markdown_to_json(md_file_name = 'info.md',
                             json_file_name = 'info.json')

    model = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    qc_bot = QCBot(model = model)
    qa_bot = QABot(model = model, qc_bot_chain = qc_bot.chain)
    with get_openai_callback() as usage_info:
        output = qa_bot.chain.invoke({'role': 'AI Developer',
                'question':'Have you deal with unstructured data? How?'
                })
        print(output)
        print(usage_info)