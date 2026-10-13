import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[str]
    verdict: str
    trace: Annotated[list[str], operator.add]

def make_plan(state: RAGState):
    return {
        "plan": ["search documents", "evaluate evidence"],
        "trace": ["Plan created"]
    }

def retrieve(state: RAGState):
    print("Search plan:", state["plan"])

    return {
        "evidence": [
            "document A: The system supports hybrid search.",
            "document B: The system supports semantic ranking."
        ],
        "trace": ["documents retrieved"]
    }

def evaluate(state: RAGState):
    enough_evidence = len(state["evidence"]) >= 2

    return {
        "verdict": "sufficient" if enough_evidence else "insufficient",
        "trace": ["Evidence evaluated"]
    }

builder = StateGraph(RAGState)

builder.add_node("make_plan", make_plan)
builder.add_node("retrieve", retrieve)
builder.add_node("evaluate", evaluate)

builder.add_edge(START, "make_plan")
builder.add_edge("make_plan", "retrieve")
builder.add_edge("retrieve", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()

result = graph.invoke({
    "question": "What search methods does the system support?",
    "plan": [],
    "evidence": [],
    "verdict": [],
    "trace": [] 
})

print("Plan: ", result["plan"])
print("Evidence: ", result["evidence"])
print("Verdict: ", result["verdict"])
print("Trace: ", result["trace"])

