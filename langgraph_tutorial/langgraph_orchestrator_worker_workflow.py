import os
import operator
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
}

missing_settings = [key for key, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

class Section(BaseModel):
    name:str = Field(..., description="Name for this section of the report")
    description: str = Field(..., description="Brief overview of the main topics and concepts to be covered in this section")


class Sections(BaseModel):
    sections: list[Section] = Field(..., description="Section of the report.")

planner = llm.with_structured_output(Sections)

class State(TypedDict):

    topic:str
    sections: NotRequired[list[Section]]
    completed_sections: Annotated[list[str], operator.add]
    final_report: NotRequired[str]

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list[str], operator.add]

def orchestrator(state: State):
    """
    Plan the report and decide what sections are needed.
    """

    plan = planner.invoke(f"""
    Create a report plan about:
    {state['topic']}
    Requirements:
    - Create 3 to 5 sections.
    - Each section should cover a different aspect.
    - Avoid overlapping sections.
    - Make the report suitable for beginners.
    """)

    return {"sections": plan.sections}

def assign_workers(state: State):
    """
    Dynamically create one worker for each section.
    """

    return [Send("worker", {"section": section, "completed_sections": []}) for section in state["sections"]]

def worker(state: WorkerState):
    """
    Write one section of the report.
    """

    section = state["section"]
    result = llm.invoke(f"""
    You are a technical writer.
    Write one section of a beginner-friendly report.

    Section title:
    {section.name}

    Section description:
    {section.description}

    Requiredments:
    - Around 100 to 150 words.
    - Use clear and simple English
    - Include practical explanations.
    - Stay focused on this section only.
    """)

    section_text = f"""
    ## {section.name}
    {result.content}
    """

    return {
        "completed_sections": [section_text]
    }

def synthesizer(state: State):
    """
    Combine all worker outputs into a final report.
    """

    sections_text = "\n".join(state["completed_sections"])

    result = llm.invoke(f"""
    You are a professional technical editor.
    
    combine the following report sections into one coherent beginner-friendly report.

    Topic:
    {state['topic']}

    Sections:
    {sections_text}

    Requiredments:
    - Add a short introduction.
    - Keep all important information.
    - Remove unnecessary repetition.
    - Add a short conclusion.
    - Use clear headings.
    """)

    return {"final_report": result.content}

builder = StateGraph(State)

builder.add_node("orchestrator", orchestrator)

builder.add_node("worker", worker)

builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")

builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])

builder.add_edge("worker", "synthesizer")

builder.add_edge("synthesizer", END)

app = builder.compile()

result = app.invoke({"topic":"Artificial Intelligence", "completed_sections": [""]})

print("\n================= Plan =================")

for section in result["sections"]:
    print(f"- {section.name}: {section.description}")

print("\n================= Worker Outputs =================")
for section in result["completed_sections"]:
    print(section)

print("\n================== Final Report =================")
print(result["final_report"])

print("\n================== Workflow =================")
print(app.get_graph().draw_mermaid())