import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field, field_validator
from typing import Union, Literal, TypedDict
from typing import Union, List
from langgraph.graph import StateGraph

class StructuredQuestionClassifierOutput(BaseModel):
    """Structuring and classifying the output of the LLM inside categories"""
    reply: Literal['quantitative', 'qualitative', 'multiple-choice'] = Field(..., description="The possible category where the question could be placed.")


class StructuredSectionClassifierOutput(BaseModel):
    """Structuring and classifying the output of the LLM inside categories"""
    reply: List[Literal['about_my_profile', 'technical_skills', 'job_preferences', 'job_experiences', 'projects', 'education', 'certifications']] = Field(..., description="The possible sections:\n'about_my_profile': personal details and overview of my professional profile.\n'technical_skills': tech stack and years of experience with them.\n'job_experiences': professional work history, including companies, responsibilities and accomplishments.\n'job_preferences': desired aspects that defines my interest for the opportunity.\n'projects': Specific projects I have worked on: project goals, role, technologies used, and the outcomes of it.\n'education': Detailed list of formal background (degrees & specializations)\n'certifications': Professional development programs, online courses, and any credentials earned.")

class StructuredQualitativeQAOutput(BaseModel):
    """Structuring the output of the LLM in Pydantic format"""
    reply: str = Field(..., description = "The answer of the question based on the provided context. If you don´t know it, return empty string")

class StructuredQuantitativeQAOutput(BaseModel):
    """Structuring the output of the LLM in Pydantic format"""
    reply: int = Field(..., description = "The answer of the question based on the provided context. If you don´t know it, return -9999")

class StructuredMultipleChoiceQAOutput(BaseModel):
    """Structuring the output of the LLM in Pydantic format"""
    reply: List[int] = Field(..., description = "A list where each element is the number that reffers the correct answer. If only one answer correct, list of 1 element. Otherwise, list of N elements where N is the amount of correct answers. If you don´t know it, return [-9999]")

    @field_validator('reply')
    def parse_my_field(cls, values):
        # Check if the values can be converted to an integer
        for value in values:
            try:
                int(float(value))
            except ValueError:
                # If it can't be converted, return it as is (string)
                return value

        return [int(float(value)) for value in values]

def convert_markdown_to_json_if_not_exist(md_file_name:str="info.md", json_file_name:str="info.json"):

    if os.path.exists(f"{WORKDIR}/resume/{json_file_name}"):
        return 


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
    convert_markdown_to_json_if_not_exist()
    print(f"The relevant sections are: {','.join(desired_sections)}")
    with open(f"{WORKDIR}/resume/info.json","r") as file:
        resume_info = json.loads(file.read())

    relevant_context = ''
    for desired_section in desired_sections:
        relevant_context += desired_section.upper() + ': ' + resume_info[desired_section] + '\n\n'

    return relevant_context

class State(TypedDict):
    question: str
    role: str
    question_type: Literal['quantitative','qualitative','multiple-choice']
    relevant_context: str
    answer: Union[str, int]

class GraphInput(TypedDict):
    question: str
    role: str

class GraphOutput(TypedDict):
    answer: Union[str, int]

class GraphConfig(TypedDict):
    classifier_model: BaseChatModel
    section_extractor_model: BaseChatModel
    qa_model: BaseChatModel

