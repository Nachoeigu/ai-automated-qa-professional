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

def defining_nodes(workflow: StateGraph):
    workflow.add_node("classifier_bot", categorize_question)
    workflow.add_node("section_extractor_bot", get_section_for_question)
    workflow.add_node("qa_bot", reply)

    return workflow

def defining_edges(workflow: StateGraph):
    workflow.add_edge("classifier_bot","section_extractor_bot")
    workflow.add_edge("section_extractor_bot","qa_bot")
    workflow.add_edge("qa_bot",END)


    return workflow


workflow = StateGraph(State, 
                      input = GraphInput,
                      output = GraphOutput,
                      config_schema = GraphConfig)

workflow.set_entry_point("classifier_bot")
workflow = defining_nodes(workflow = workflow)
workflow = defining_edges(workflow = workflow)

app = workflow.compile()


if __name__ == '__main__':
    model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)

    output = app.invoke({'question':"Years of experience with Docker?", "role": "AI Engineer"}, 
            config = {"configurable":{"classifier_model": model,
                                        "section_extractor_model": model,
                                        "qa_model": model,
                                        "thread_id":1233
                                        }
                        }
            )

    print(output)