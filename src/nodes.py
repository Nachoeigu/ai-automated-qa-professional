import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from src.utils import State
from src.model import QABot, QuestionClassifierBot, SectionClassifierBot
from src.utils import extracting_relevant_context_from_resume, GraphConfig

def categorize_question(state: State, config: GraphConfig) -> State:
    """
    This function categorizes the question as quantitative, qualitative or multiple-choice
    """
    bot = QuestionClassifierBot(model = config['configurable']['classifier_model'])
    classification = bot.model.invoke(state['question'])
    
    return {'question_type': classification['reply']}

def get_section_for_question(state: State, config: GraphConfig) -> State:
    """
    This function extracts the section/s of your resume where the answer of the question could be present.
    """
    bot = SectionClassifierBot(model = config['configurable']['section_extractor_model'])
    answer = bot.model.invoke(state['question'])
    
    return {'sections': answer['reply']}

def reply(state: State, config: GraphConfig) -> State:
    """
    This function answers the initial question with the needed context
    """
    resume_info = extracting_relevant_context_from_resume(state['sections'])
    bot = QABot(model = config['configurable']['qa_model'], question_type = state['question_type'])

    answer = bot(question=state['question'],
                role=state['role'],
                resume_info=resume_info
            )
    
    return {'answer': answer['reply']}