import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from deepagents import create_deep_agent

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT
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
def get_weather(city: str) -> str:
    """Get the weather for a given city."""

    weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C",
        "San Francisco": "Foggy, 16°C"
    }

    city = city.strip().title()

    data = weather_data.get(city, "Weather data not available for this city.")

    return data

agent = create_deep_agent(
    model=llm, 
    tools=[get_weather],
    system_prompt="""
    You are a helpful weather assistant.

    Use the get_weather tool whenever the user asks about the weather.

    After receiving the tool result, explain the answer clearly.
    """
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in New York?"
            }
        ]
    }
)

for index, message in enumerate(result["messages"], start=1):
    print(f"\nMessage {index}")
    print(f"Type:{type(message).__name__}")

    if message.content:
        print("Content:")
        print(message.content)

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")
        for tool_call in tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])

print("\n============= FINAL ANSWER =============")
print(result["messages"][-1].content)

