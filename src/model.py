import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from operator import itemgetter
import json
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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

        self.__developing_template()

    
    def __developing_template(self):
        self.prompt_template = self.system_prompt+"\nProblem to solve:\nA candidate is applying for the role: `{role}´\nIt needs to answer this: `{question}´\nUse the following context to formulate your response:\n```{resume_info}```"


    def __call__(self, question, role, resume_info):
        prompt = self.prompt_template.format(question=question, role=role, resume_info=resume_info)
        output = self.model.invoke(prompt)
        return output
                                

class QuestionClassifierBot:

    def __init__(self, model):
        self.model = model.with_structured_output(StructuredQuestionClassifierOutput)
        self.__developing_template()
        self.chain = self.__developing_chain()  

    def __developing_template(self):
        self.prompt_template = PromptTemplate(
                    template="{system_prompt}\n`{question}´",
                    input_variables=["question"],
                    partial_variables={
                                    "system_prompt": QC_SYSTEM_PROMPT,
                                    },
                )
        
    def __developing_chain(self):
        return itemgetter("question") \
                    | self.prompt_template \
                        | self.model 


class SectionClassifierBot:
    def __init__(self, model):
        self.model = model.with_structured_output(StructuredSectionClassifierOutput)
        self.__developing_template()
        self.chain = self.__developing_chain()

    def __developing_template(self):
        self.prompt_template = PromptTemplate(
                    template="{system_prompt}\n`{question}´",
                    input_variables=["question"],
                    partial_variables={
                                    "system_prompt": SC_SYSTEM_PROMPT,
                                    },
                )
        
    def __developing_chain(self):
        return itemgetter("question") \
                    | self.prompt_template \
                        | self.model \
                            | RunnableLambda(lambda qc_answer: extracting_relevant_context_from_resume(qc_answer.reply))

