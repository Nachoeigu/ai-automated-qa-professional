import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import json
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from src.utils import convert_markdown_to_json, extracting_relevant_context_from_resume, StructuredQAOutput, StructuredClassifierOutput
from src.constants import QA_SYSTEM_PROMPT, QC_SYSTEM_PROMPT

class QABot:

    def __init__(self, model, qc_bot_chain):
        self.model = ChatOpenAI(model="gpt-4o-mini", 
                    temperature = 0)
        self.parser = PydanticOutputParser(pydantic_object=StructuredQAOutput)
        self.qc_bot_chain = qc_bot_chain
        self.__developing_template()
        self.chain = self.__developing_chain()

    
    def __developing_template(self):
        self.prompt_template = PromptTemplate(
                    template="{system_prompt}\nProblem to solve:\nA candidate is applying for the role: `{role}´\nIt needs to answer this: `{question}´\nUse the following context to formulate your response:```{resume_info}```\n\n Follow these format instructions: ```{format_instructions}```",
                    input_variables=["role","question","resume_info"],
                    partial_variables={"format_instructions": self.parser.get_format_instructions(),
                                    "system_prompt": QA_SYSTEM_PROMPT,
                                    },
                )

    def __developing_chain(self):
        return {'role': itemgetter("role"), 'question': itemgetter("question"), 'resume_info': self.qc_bot_chain} \
                    | self.prompt_template \
                            | self.model \
                                | {
                                    'output': self.parser \
                                                | RunnableLambda(lambda output: json.loads(output.json())), 
                                    'token_usage': RunnableLambda(lambda input_data: input_data.usage_metadata)
                                }
        
class QCBot:
    def __init__(self, model):
        self.model = ChatOpenAI(model="gpt-4o-mini", 
                    temperature = 0)
        self.parser = PydanticOutputParser(pydantic_object=StructuredClassifierOutput)
        self.__developing_template()
        self.chain = self.__developing_chain()

    def __developing_template(self):
        self.prompt_template = PromptTemplate(
                    template="{system_prompt}\n`{question}´\n\n Follow these format instructions: ```{format_instructions}```",
                    input_variables=["question"],
                    partial_variables={"format_instructions": self.parser.get_format_instructions(),
                                    "system_prompt": QC_SYSTEM_PROMPT,
                                    },
                )
        
    def __developing_chain(self):
        return itemgetter("question") \
                    | self.prompt_template \
                        | self.model \
                            | self.parser \
                                | RunnableLambda(lambda qc_answer: extracting_relevant_context_from_resume(qc_answer.reply))

