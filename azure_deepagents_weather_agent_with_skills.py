import os
import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from pathlib import Path
from rich import print as pprint

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

BASE_DIR = Path(__file__).resolve().parent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""

    weather_data = {
        "New York": "Sunny",
        "Los Angeles": "Cloudy",
        "Chicago": "Rainy"
    }

    city = city.strip().title()

    data = weather_data.get(city, "Weather data not available")

    return data

agent = create_deep_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
    You are a helpful weather assistant.

    Use the get_weather tool whenever th user asks about weather.
    Follow the project memory and relevant skills.
    """,
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    backend=FilesystemBackend(
        root_dir=str(BASE_DIR),
        virtual_mode=True
    )
)

result = agent.invoke({"messages":[{"role":"user","content":"What's the weather in New York?"}]}, config={"configurable":{"thread_id":"weather-demo"}})

print(result["messages"][-1].content)