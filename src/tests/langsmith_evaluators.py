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
from src.tests.functions import checking_if_dataset_is_already_created
from langchain_core.language_models.chat_models import BaseChatModel


class LangsmithEvaluator:

    def __init__(self, evaluator_model: BaseChatModel):
        self.langsmith_client = Client()
        checking_if_dataset_is_already_created(self.langsmith_client)
        self.qa_evaluator = self.__config_evaluator(evaluator_model)
        
    def __config_evaluator(self, evaluator_model):
        evaluator_prompt = PromptTemplate(
            input_variables=["input", "reference", "prediction"], template=EVALUATOR_SYSTEM_PROMPT
        )
        return LangChainStringEvaluator("qa", 
                config={"llm": evaluator_model, "prompt": evaluator_prompt}
                )  

    def set_testing_models(self, qc_model: BaseChatModel, qa_model: BaseChatModel, desired_metadata:dict, experiment_prefix_name:str):
        self.desired_metadata = desired_metadata
        self.experiment_prefix_name = experiment_prefix_name
        self.qc_bot = QCBot(model = qc_model)
        self.qa_bot = QABot(model = qa_model, qc_bot_chain = self.qc_bot.chain)
        
    def generate_testing_outputs(self):
        evaluate(
            lambda input:self.qa_bot.chain.invoke({'role': '', 'question':input['question']}),
            data="ai_qa_professional_dataset", 
            evaluators=[self.qa_evaluator],
            metadata=self.desired_metadata, #{"username": "Nachito"}
            experiment_prefix= self.experiment_prefix_name #'correctness_in_gpt4o_mini'
        )

        print("Check the results of the testing on Langsmith site...")
         


if __name__ == '__main__':
    #Here you put the best model possible:
    evaluator_model = ChatOpenAI(model = 'chatgpt-4o-latest', temperature = 0)
    #The models you want to test:
    qc_model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature = 0)
    qa_model = ChatGroq(model = 'llama3-groq-70b-8192-tool-use-preview', temperature = 0)

    testing_bot = LangsmithEvaluator(evaluator_model = evaluator_model)
    testing_bot.set_testing_models(qc_model = qc_model, 
                                   qa_model = qa_model, 
                                   desired_metadata={'username': 'Nachiiin'},
                                   experiment_prefix_name='correctness_llama_3')
    testing_bot.generate_testing_outputs()
