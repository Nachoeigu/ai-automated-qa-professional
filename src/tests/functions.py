import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import json
from langsmith import Client



def checking_if_dataset_is_already_created(langsmith_client:Client):
    """
    It checks if the dataset is already in Langsmith, if not, it creates the dataset. Otherwise, it doesn´t make anything
    """
    if "ai_qa_professional_dataset" not in [dataset.name for dataset in langsmith_client.list_datasets()]:
        langsmith_client.create_dataset(
            dataset_name = 'ai_qa_professional_dataset',
            description = 'QA pairs about a professional profile'
        )
        with open(f"{WORKDIR}/src/tests/dataset.json", "r") as file:
            dataset = json.loads(file.read())

        questions = [{'question': case['question']} for case in dataset]
        answers = [{'answer': case['answer']} for case in dataset]

        langsmith_client.create_examples(
            inputs = questions,
            outputs = answers,
            dataset_name="ai_qa_professional_dataset"
        )
