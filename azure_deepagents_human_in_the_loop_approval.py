import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "TAVILY_API_KEY": TAVILY_API_KEY
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

tavily_search = TavilySearch(
    max_results=1,
    topic="finance",
    api_key=TAVILY_API_KEY
)

@tool
def remove_file(path: str) -> str:
    """Simulate deleting a file."""
    return f"[SIMULATION] Deleted {path}"

@tool
def fetch_file(path: str) -> str:
    """Simulate fetching a file."""
    return f"[SIMULATION] Contents of {path}"

@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email."""

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    tools=[remove_file, fetch_file, notify_email],
    interrupt_on={
        "remove_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "fetch_file": False,
        "notify_email": {"allowed_decisions": ["approve", "reject"]}
    },
    checkpointer=checkpointer,
    system_prompt="""
    When the user asks to delete a file, call the custom remove_file tool.
    Do not use any built-in filesystem deletion tool.

    when the user asks to read a file, call fetch_file.
    When the user asks to send an email, call notify_email.
    """
)

config = {
    "configurable": {"thread_id":"hitl_practice-001"}
}

result = agent.invoke({"messages":[{"role":"user", "content": "Please delete the file /tmp/old_report.txt. Before sending any email, ask me for approval."}]}, config=config)

interrupts = result.get("__interrupt__", [])

if not interrupts:
    print("Agent did not pause. Check the reqeust and tool call.")
else:
    print("Approval required:")
    print(interrupts)

    decision = input("\nApprove or reject? (a/r): ").strip().lower()

    if decision == "a":
        resume_command = Command(resume={"decisions": [{"type": "approve"}]})
    elif decision == "r":
        resume_command = Command(resume={"decisions": [{"type": "reject", "message": "The user rejected this action."}]})
    else:
        raise ValueError("Please enter 'a' to approve or 'r' to reject.")

    final_result = agent.invoke(resume_command, config=config)

    print("\nMessages after resume:")
    for message in final_result["messages"]:
        print(type(message).__name__,":", message.content)