import operator
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch

def classify_intent(state: dict):
    question = state["question"].lower()

    if "error" in question or "bug" in question:
        intent = "bug"
    else:
        intent = "doc"

    return {
        **state,
        "Intent": intent
    }

def search_docs(state: dict):
    return {
        **state,
        "context": "Documentation: Hybrid Search combines keyword and vector search."
    }

def search_bug(state: dict):
    return {
        **state,
        "context": "Bug Tracker: No known issue was found."
    }

def generate_reply(state:dict):
    return {
        **state,
        "reply": f"Intent: {state['Intent']}\nResult: {state['context']}"
    }


classifier = RunnableLambda(classify_intent)

router = RunnableBranch((lambda state: state["Intent"] == "bug", RunnableLambda(search_bug)), RunnableLambda(search_docs))

reply = RunnableLambda(generate_reply)

workflow = classifier | router | reply


result = workflow.invoke({"question": "How does hybrid search work?"})

print(result["reply"])