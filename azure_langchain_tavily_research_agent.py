import os, json

from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
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

missing_settings = [name for name, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

tavily_search = TavilySearch(max_results=5, topic="finance", api_key=TAVILY_API_KEY)

agent = create_agent(
    model=llm, 
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. Use the Tavily search tool when the user asks for current or web-based information. Summarize the search results clearly."
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "現在台股台積電股價是多少?"
            }
        ]
    }
)


for index, message in enumerate(result["messages"]):
    print(f"Message {index + 1}:")
    print("Type:", type(message).__name__)
    print("Content:", message.content)

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")
        for tool_call in tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])
            print("Tool call ID:", tool_call["id"])


last_message = result["messages"][-1]

print("Final Answer:")
print(last_message.content)



