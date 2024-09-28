import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import  extracting_relevant_context_from_resume, StructuredMultipleChoiceQAOutput, StructuredQuantitativeQAOutput, StructuredQualitativeQAOutput, StructuredSectionClassifierOutput, StructuredQuestionClassifierOutput
from constants import QA_SYSTEM_PROMPT, QC_SYSTEM_PROMPT, SC_SYSTEM_PROMPT

class QABot:

    def __init__(self, model, question_type: str):
        if question_type == 'qualitative':
            self.model = model.with_structured_output(StructuredQualitativeQAOutput)
            self.system_prompt = QA_SYSTEM_PROMPT.format(CASE_DONT_KNOW_ANSWER="empty string")
        elif question_type == 'quantitative':
            self.model = model.with_structured_output(StructuredQuantitativeQAOutput)
            self.system_prompt = QA_SYSTEM_PROMPT.format(CASE_DONT_KNOW_ANSWER="-9999")            
        else:
            self.model = model.with_structured_output(StructuredMultipleChoiceQAOutput)
            self.system_prompt = QA_SYSTEM_PROMPT.format(CASE_DONT_KNOW_ANSWER="[-9999]")            

        self.prompt_template = "\nProblem to solve:\nA candidate is applying for the role: `{role}´\nIt needs to answer this: `{question}´\nUse the following context to formulate your response:\n```{resume_info}```"


    def __call__(self, question, role, resume_info):
        messages = [SystemMessage(content = self.system_prompt),
                    HumanMessage(content = self.prompt_template.format(role=role,question=question,resume_info=resume_info))]
                
        return self.model.invoke(messages)
        

class QuestionClassifierBot:

    def __init__(self, model):
        self.model = model.with_structured_output(StructuredQuestionClassifierOutput)
        self.system_prompt = QC_SYSTEM_PROMPT

    def __call__(self, question: str):
        messages = [SystemMessage(content=self.system_prompt),
                   HumanMessage(content=question)]

        return self.model.invoke(messages)



class SectionClassifierBot:
    def __init__(self, model):
        self.model = model.with_structured_output(StructuredSectionClassifierOutput)
        self.system_prompt = SC_SYSTEM_PROMPT
        self.chain = self.model | RunnableLambda(lambda qc_answer: extracting_relevant_context_from_resume(qc_answer['reply']))

    def __call__(self, question:str):
        messages = [SystemMessage(content = self.system_prompt),
                    HumanMessage(content = question)]
        
        return self.chain.invoke(messages)

