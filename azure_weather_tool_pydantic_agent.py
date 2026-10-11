import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

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

missing_settings = [name for name, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {", ".join(missing_settings)}")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)


class WeatherInput(BaseModel):
    city: str  = Field(..., description="Name of the city to get the weather for")

class WeatherOutput(BaseModel):
    city: str = Field(..., description="Name of the city for which the weather is provided")
    condition: str = Field(..., description="Weather condition in the specified city")
    temperature: float = Field(..., description="Temperature in the specified city")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    This is a demo tool with mock data
    """

    weather_data = {
        "Tokyo": {"condition": "Cloudy", "temperature": 22.5},
        "Osaka": {"condition": "Sunny", "temperature": 25.0},
        "Taipei": {"condition": "Rainy", "temperature": 20.0}
    }

    data = weather_data.get(city, {"condition": "Unknown", "temperature": 0})

    result = WeatherOutput(city=city, condition=data["condition"], temperature=data["temperature"])


    return result.model_dump_json()

print(f"Tool name: {get_weather.name}")
print(f"Tool description: {get_weather.description}")
print(f"Tool schema: {get_weather.get_input_schema().model_json_schema()}")


agent = create_agent(model=llm, tools=[get_weather], system_prompt="You are a helpful weather assistant. Use the get_weather tool when the user asks about weather. After receiving the tool result, explain the weather clearly.")

result = agent.invoke({"messages":[{"role":"user","content":"What's the weather in Tokyo?"}]})

for index, message in enumerate(result["messages"], start=1):
    print(f"Message {index}")
    print(f"Type:{type(message).__name__}")

    if message.content:
        print(f"Content:{message.content}")

    if getattr(message, "tool_calls", None):
        print(f"Tool Calls")

        for tool_call in message.tool_calls:
            print(f"Tool Call: {tool_call.get('name')}")
            print(f"Tool Call Arguments: {tool_call.get('args')}")

print("Final Answer:")
print(result.get("messages")[-1].content)