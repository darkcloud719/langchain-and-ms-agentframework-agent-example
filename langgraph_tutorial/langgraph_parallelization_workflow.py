import os
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired
from langchain_openai import AzureChatOpenAI
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

# Define LangGraph State

class State(TypedDict):
    topic: str
    explanation: NotRequired[str]
    advantages: NotRequired[str]
    limitations: NotRequired[str]
    examples: NotRequired[str]
    combined_output: NotRequired[str]

# Parallel Node 1

def generate_explanation(state: State):
    """
    Explain the topic to a beginner.
    """
    msg = llm.invoke(f"""
    Explain "{state['topic']}" to a beginner.
    Requirements:
    - Use simple language.
    - Keep it under 100 words.
    """)

    return {"explanation": msg.content}

# Parallel Node 2
def generate_advantages(state: State):
    """
    Generate the advantages of the topic.
    """

    msg = llm.invoke(f"""
    What are the main advantages of "{state['topic']}"?

    Requirements:
    - Give exactly 3 points.
    - Keep each point concise
    """)

    return {
        "advantages": msg.content
    }

# Parallel Node 3
def generate_limitations(state: State):
    """
    Generate the limitations of the topic.
    """

    msg = llm.invoke(f"""
    What are the main limitations of "{state['topic']}"?

    Requirements:
    - Give exactly 3 points.
    - Keep each point concise.
    """)

    return {
        "limitations": msg.content
    }

# Parallel Node 4
def generate_examples(state: State):
    """
    Generate real-world examples.
    """

    msg = llm.invoke(f"""
    Give 3 real-world use cases of "{state['topic']}".

    Requirements:
    - Use practical examples.
    - Keep each example concise.
    """)

    return {
        "examples": msg.content
    }

def aggregator(state: State):
    """
    Combine all parallel results into one output.
    """

    combined = f"""
    ========================
    Topic: {state['topic']}
    ========================
    Explanation: {state['explanation']}

    Advantages: {state['advantages']}

    Limitations: {state['limitations']}

    Real-World Examples: {state['examples']}
    """

    return {"combined_output": combined}

builder = StateGraph(State)

builder.add_node("generate_explanation", generate_explanation)
builder.add_node("generate_advantages", generate_advantages)
builder.add_node("generate_limitations", generate_limitations)
builder.add_node("generate_examples", generate_examples)
builder.add_node("aggregator", aggregator)

builder.add_edge(START,"generate_explanation")
builder.add_edge(START,"generate_advantages")
builder.add_edge(START, "generate_limitations")
builder.add_edge(START, "generate_examples")

builder.add_edge("generate_explanation", "aggregator")
builder.add_edge("generate_advantages", "aggregator")
builder.add_edge("generate_limitations", "aggregator")
builder.add_edge("generate_examples", "aggregator")

builder.add_edge("aggregator", END)

app = builder.compile()

result = app.invoke({"topic": "Artificial Intelligence"})

print("\n================= Final Output =======================")
print(result["combined_output"])

print("\n================= Workflow =======================")
print(app.get_graph().draw_mermaid())
