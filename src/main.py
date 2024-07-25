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

if __name__ == '__main__':
    convert_markdown_to_json(md_file_name = 'long_version.md',
                             json_file_name = 'info.json')
    qc_parser = PydanticOutputParser(pydantic_object=StructuredClassifierOutput)
    qc_custom_template = PromptTemplate(
                template="{system_prompt}\n`{question}´\n\n Follow these format instructions: ```{format_instructions}```",
                input_variables=["question"],
                partial_variables={"format_instructions": qc_parser.get_format_instructions(),
                                "system_prompt": QC_SYSTEM_PROMPT,
                                },
            )



    qa_parser = PydanticOutputParser(pydantic_object=StructuredQAOutput)

    qa_custom_template = PromptTemplate(
                template="{system_prompt}\Problem to solve:\nA candidate is applying for the role: `{role}´\nIt needs to answer this: `{question}´\nUse the following context to formulate your response:```{resume_info}```\n\n Follow these format instructions: ```{format_instructions}```",
                input_variables=["role","question","resume_info"],
                partial_variables={"format_instructions": qa_parser.get_format_instructions(),
                                "system_prompt": QA_SYSTEM_PROMPT,
                                },
            )

    model = ChatOpenAI(model="gpt-4o-mini", 
                        temperature = 0)
    qc_chain = itemgetter("question") \
                    | qc_custom_template \
                        | model \
                            | qc_parser \
                                | RunnableLambda(lambda qc_answer: extracting_relevant_context_from_resume(qc_answer.reply))
    chain = {'role': itemgetter("role"), 'question': itemgetter("question"), 'resume_info': qc_chain} \
                | qa_custom_template \
                        | model \
                            | {
                                'output': qa_parser \
                                            | RunnableLambda(lambda output: json.loads(output.json())), 
                                'token_usage': RunnableLambda(lambda input_data: input_data.usage_metadata)
                            }

    with get_openai_callback() as usage_info:
        output = chain.invoke({'role': 'AI Developer',
                'question':'Are you open for hybrid job?'
                })
        #output = qc_chain.invoke({"question":"Explain one project you developed end to end and you are proud of it"})
        print(output)
        print(usage_info)