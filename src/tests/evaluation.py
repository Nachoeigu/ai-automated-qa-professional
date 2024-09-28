import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langgraph.graph import StateGraph, END
from src.utils import State, GraphInput, GraphOutput, GraphConfig
from langchain_groq.chat_models import ChatGroq
from src.nodes import *
from src.agent import app
import json
from pydantic import BaseModel, Field
from constants import EVALUATOR_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def get_testing_dataset():
    with open(f"{WORKDIR}/src/tests/dataset.json","r") as file:
        testing_data = json.loads(file.read())

    return testing_data

class StructuredEvaluatorOutput(BaseModel):
    """
    Structured output of the Evaluator LLM
    """
    is_correct: bool = Field(description = "If the case you are analyzing is correct, place True. If the answer is wrong, place False.")


class Evaluator:

    def __init__(self, model):
        self.model = model.with_structured_output(StructuredEvaluatorOutput)
        self.system_prompt = EVALUATOR_SYSTEM_PROMPT

    def __call__(self, query, real_answer, model_answer):
        messages = [SystemMessage(content = self.system_prompt),
                    HumanMessage(content = f"You are grading the following question: {query}\nHere is the real answer: {real_answer}\nYou are grading the following predicted answer: {model_answer}")]
        
        return self.model.invoke(messages)


if __name__ == '__main__':
    model = ChatGroq(model="llama3-groq-8b-8192-tool-use-preview", temperature=0)
    evaluator_model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    testing_data = get_testing_dataset()
    evaluator_bot = Evaluator(model = evaluator_model)
    config = {"configurable":{
                "classifier_model": model,
                "section_extractor_model": model,
                "qa_model": model,
                "thread_id":1233
            }
        }
    
    good_prediction = 0
    bad_prediction = 0
    for case in testing_data:
        real_value = case['answer']
        model_output = app.invoke({'question':case['question'], "role": "AI Engineer"}, 
                config = config
                )
        predicted_value = model_output['answer']
        #If the case is quantitative or multiple-choice we can measure the error with some logics
        if isinstance(predicted_value, str) == False:
            if real_value == predicted_value:
                good_prediction += 1
            else:
                bad_prediction += 1

        #Otherwise, we use an llm as evaluator
        else:
            evaluator_output = evaluator_bot(query = case['question'], real_answer = real_value, model_answer = predicted_value)
            
            if evaluator_output['is_correct'] == True:
                good_prediction += 1
            else:
                bad_prediction += 1
    

    print(f"% of Correct Predictions over Total cases: {good_prediction / len(testing_data)}")







