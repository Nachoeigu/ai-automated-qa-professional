import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field, validator, conlist
from typing import Union, Literal
from langchain_core.pydantic_v1 import BaseModel, Field, conlist, constr
from typing import Union, List


class StructuredClassifierOutput(BaseModel):
    """Structuring and classifying the output of the LLM inside categories"""
    reply: List[Literal['about_my_profile', 'technical_skills', 'job_preferences', 'job_experiences', 'projects', 'education', 'certifications']] = Field(..., description="The possible sections:\n'about_my_profile': personal details and overview of my professional profile.\n'technical_skills': tech stack and years of experience with them.\n'job_experiences': professional work history, including companies, responsibilities and accomplishments.\n'job_preferences': desired aspects that defines my interest for the opportunity.\n'projects': Specific projects I have worked on: project goals, my role, technologies used, and the outcomes of it.\n'education': Detailed list of formal background (degrees & specializations)\n'certifications': Professional development programs, online courses, and any credentials earned.")

class StructuredQAOutput(BaseModel):
    """Structuring the output of the LLM in Pydantic format"""
    reply: Union[str, int] = Field(..., description = "The answer of the provided question. Three possible ways: If quantitative question, reply as integer; if multiple choice question with options, reply with index enumeration of the correct option. If qualitative question, reply as string.")

    @validator('reply')
    def parse_my_field(cls, v):
        # Check if the value can be converted to an integer
        try:
            return int(float(v))
        except ValueError:
            # If it can't be converted, return it as is (string)
            return v


def convert_markdown_to_json(md_file_name:str="info.md", json_file_name:str="info.json"):
    with open(f"{WORKDIR}/resume/{md_file_name}", "r") as file:
        markdown_document = file.read()
    headers_to_split_on = [
        ("#", "section"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    output_json = {}
    for md_part in md_header_splits:
        output_json[md_part.metadata['section'].replace(' ','_').lower()] = md_part.page_content

    with open(f"{WORKDIR}/resume/{json_file_name}","w") as file:
        json.dump(output_json, file,ensure_ascii=False)

def extracting_relevant_context_from_resume(desired_sections:list):
    print(f"The relevant sections are: {','.join(desired_sections)}")
    with open(f"{WORKDIR}/resume/info.json","r") as file:
        resume_info = json.loads(file.read())

    relevant_context = ''
    for desired_section in desired_sections:
        relevant_context += desired_section.upper() + ': ' + resume_info[desired_section] + '\n\n'

    return relevant_context


if __name__ == '__main__':
    print(StructuredClassifierOutput(reply = ['job_experiences','technical_skills']))
