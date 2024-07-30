import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_openai import ChatOpenAI
from src.model import QABot, QCBot
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langsmith.evaluation import LangChainStringEvaluator
from src.constants import EVALUATOR_SYSTEM_PROMPT
import json
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


langsmith_client = Client()
if "ai_qa_professional_dataset" not in [dataset.name for dataset in langsmith_client.list_datasets()]:
    langsmith_client.create_dataset(
        dataset_name = 'ai_qa_professional_dataset',
        description = 'QA pairs about a professional profile'
    )
    with open(f"{WORKDIR}/src/tests/dataset.json", "r") as file:
        dataset = json.loads(file.read())

    questions = [{'question': case['question']} for case in dataset]
    answers = [{'answer': case['answer']} for case in dataset]

    langsmith_client.create_examples(
        inputs = questions,
        outputs = answers,
        dataset_name="ai_qa_professional_dataset"
    )

evaluator_model = ChatOpenAI(model='gpt-4o', temperature=0)

evaluator_prompt = PromptTemplate(
    input_variables=["input", "reference", "prediction"], template=EVALUATOR_SYSTEM_PROMPT
)

qa_evaluator = LangChainStringEvaluator("qa", 
                                        config={"llm": evaluator_model, "prompt": evaluator_prompt})  

def predict_with_llm(input):
    model = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    qc_bot = QCBot(model = model)
    qa_bot = QABot(model = model, qc_bot_chain = qc_bot.chain)
    output = qa_bot.chain.invoke({'role': 'AI Developer',
            'question':input['question']
            })

    return output


evaluate(
    predict_with_llm,
    data="ai_qa_professional_dataset", 
    evaluators=[qa_evaluator],
    metadata={"username": "Nachito"},
    experiment_prefix='correctness_in_gpt4o_mini'
)
