import os
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

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

class Evaluation(BaseModel):
    grade: Literal["pass", "fail"] = Field(..., description="Decide whether the explanation is good enough.")
    feedback: str = Field(..., description="If the explanation fails, explain how to improve it. If it passes, briefly explain why.")

evaluator = llm.with_structured_output(Evaluation)

class State(TypedDict):
    topic: str
    explanation: NotRequired[str]
    grade: NotRequired[str]
    feedback: NotRequired[str]
    iteration: int

def generator(state: State):
    """
    Generate or improve the technical explanation.
    """

    if not state.get("feedback"):
        prompt = f"""
        Explain "{state['topic']}" to a beginner.

        Requirements:
        - Use simple English.
        - Keep it under 150 words.
        - Explain the core concept accurately.
        - Give one practical example.
        """
    else:
        prompt = f"""
        Rewrite the explanation about "{state['topic']}".

        Previous explanation:
        {state['explanation']}

        Evaluator feedback:
        {state['feedback']}

        Requirements:
        - Fix the problems identified by the evaluator.
        - Use simple English.
        - Keep it under 150 words.
        - Explain the core concept accurately.
        - Give one practical example.
        """

    result = llm.invoke([
        SystemMessage(content="You are an AI technology instructor. You job is to explain technical concepts clearly to beginners. Be accurate, concise, and easy to understand."),
        HumanMessage(content=prompt)
    ])
    return {
        "explanation": result.content,
        "iteration": state["iteration"] + 1
    }

def evaluate(state: State):
    """
    Evaluate the generated explanation.
    """

    result = evaluator.invoke([
        SystemMessage(content="""
        You are a strict technical reviewer.
        Evaluate whether an explanation is suitable for a beginner.

        An explanation passes only if it is:

        1. Technically correct
        2. Easy to understand
        3. Concise
        4. Focused on the topic
        5. Includes a useful practical example
        """),
        HumanMessage(content=f"""
        Topic: {state['topic']}
        Explanation: {state['explanation']}
        Evaluate this explanation.

        """)
    ])

    return {
        "grade": result.grade,
        "feedback": result.feedback
    }

def route_evaluation(state: State):
    """
    Decide whether to finish or improve the answer.
    """

    if state["grade"] == "pass":
        return "accepted"

    if state["iteration"] >= 3:
        return "max_iterations"

    return "retry"


builder = StateGraph(State)

builder.add_node("generator", generator)
builder.add_node("evaluator", evaluate)

builder.add_edge(START,"generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator", route_evaluation, 
    {
        "accepted": END,
        "retry": "generator",
        "max_iterations": END
    })

app = builder.compile()

result = app.invoke(
    {
        "topic": "Hybrid Search",
        "iteration": 0
    }
)

print("\n===================== Explanation =====================\n")
print(result["explanation"])
print("\n===================== Evaluation =====================\n")
print("Grade:", result["grade"])
print("Feedback:", result["feedback"])
print("Iterations:", result["iteration"])
print("\n==================== Workflow =====================\n")
print(app.get_graph().draw_mermaid())
