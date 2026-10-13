import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from langgraph.graph import END, START, StateGraph

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

@tool
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a:int, b:int) -> float:
    """Divide two numbers"""
    return a / b

@tool
def subtract(a:int, b:int) -> int:
    """Subtract two numbers"""
    return a - b

tools = [add, subtract, multiply, divide]

tools_by_name = {current_tool.name: current_tool for current_tool in tools}

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: AgentState) -> dict:
    system_message = SystemMessage(content="You are a helpful arithmetic assistant. Use the provided tools to perform calculations.")

    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {
        "messages": [response],
        "llm_calls": state["llm_calls"] + 1
    }

def tool_node(state: AgentState) -> dict:
    tool_messages = []

    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        selected_tool = tools_by_name[tool_name]
        observation = selected_tool.invoke(tool_args)

        tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")

builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

builder.add_edge("tool_node", "llm_call")

agent = builder.compile()

def main() -> None:
    question = input("請輸入算式問題，E.G. 3 and 4:").strip()

    if not question:
        question = "Add 3 and 4."

    result = agent.invoke({"messages": [HumanMessage(content=question)], "llm_calls": 0})

    print("Result:")

    for message in result["messages"]:
        message.pretty_print()

    print(f"\nLLM calls: {result['llm_calls']}")

if __name__ == "__main__":
    main()

