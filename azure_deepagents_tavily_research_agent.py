import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

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
    max_results=5,
    topic="finance",
    api_key=TAVILY_API_KEY
)

research_subagent = {
    "name": "research_agent",
    "description": "Researches current financial information and other web-based questions using Tavily search.",
    "system_prompt": """
    You are a careful research assistant.
    Use the Tavily search tool when you need current or web-based information.
    When researching:
    1. Search for relevant information.
    2. Summarize the findings clearly.
    3. Include source URLs when available.
    4. State clearly when the information may be incomplete or outdated.
    """,
    "tools": [tavily_search],
    "model": llm
}

agent = create_deep_agent(
    model=llm,
    subagents=[research_subagent],
    system_prompt="""
    You are a helpful research assistant.

    Delegate questions to the research-agent when they require:
    - current information
    - web searches
    - financial or market information
    - fact checking or research from online sources

    for simple questions that do not require web research, answer directly.

    After receiving the research-agent's findings, summarize them clearly.
    Include relevant source URLs in your final answer.
    """
)

result = agent.invoke({"messages": [{"role": "user", "content": "現在台股台積電股價是多少?"}]})

for index, message in enumerate(result["messages"], start=1):
    print(f"Message {index}:")
    print(f"Type: {type(message).__name__}")
    print(f"Content: {message.content}")

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")

        for tool_call in tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])
            print("Tool call ID:", tool_call["id"])

    print("-"*60)

last_message = result["messages"][-1]

print("Final Answer:")
print(last_message.content)



