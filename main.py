import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from operator import itemgetter
from langchain.prompts import PromptTemplate
from constants import SYSTEM_PROMPT
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import RunnableLambda
from typing import Union
import json
from langchain_community.callbacks import get_openai_callback

from langchain_openai import ChatOpenAI


with open(f"{WORKDIR}/resume/info.md","r") as file:
    resume_info = file.read()

class StructuredLLMOutput(BaseModel):
    """Structuring the output of the LLM in Pydantic format"""
    reply: Union[str, int] = Field(..., description = "The answer of the provided question. If qualitative question, reply as string; if quantitative question reply as integer.")

    @validator('reply')
    def parse_my_field(cls, v):
        # Check if the value can be converted to an integer
        try:
            return int(v)
        except ValueError:
            # If it can't be converted, return it as is (string)
            return v


parser = PydanticOutputParser(pydantic_object=StructuredLLMOutput)


custom_template = PromptTemplate(
            template="{system_prompt}\Problem to solve:\nA candidate is applying for the role: `{role}´\nIt needs to answer this: `{question}´\nUse the following context to formulate your response:```{resume_info}```\n\n Follow these format instructions: ```{format_instructions}```",
            input_variables=["role","question"],
            partial_variables={"format_instructions": parser.get_format_instructions(),
                               "system_prompt": SYSTEM_PROMPT,
                               "resume_info": resume_info
                               },
        )

model = ChatOpenAI(model="gpt-4o-mini", 
                    temperature = 0)

chain = custom_template \
                | model \
                    | {
                        'output': parser \
                                    | RunnableLambda(lambda output: json.loads(output.json())), 
                        'token_usage': RunnableLambda(lambda input_data: input_data.usage_metadata)
                    }

with get_openai_callback() as usage_info:
    output = chain.invoke({'role': 'AI Developer',
              'question':'Are you open for presencial job?'
              })
    print(output)
    print(usage_info)