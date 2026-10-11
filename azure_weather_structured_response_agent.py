import os, json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
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

missing_settings = [name for name, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

class WeatherInput(BaseModel):
    city: str = Field(..., description="The name of the city to get the weather for.")

class WeatherOutput(BaseModel):
    city: str = Field(..., description="The name of the city for which the weather is provided.")
    condition: str = Field(..., description="The weather condition in the specified city.")
    temperature_c: int = Field(..., description="The temperature in Celsius in the specified city.")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
# def get_weather(city: str) -> WeatherOutput:
    """
    Get the current weather for the specified city.
    Args:
        city (str): The name of the city to get the weather for.
    Returns:
        str: A string describing the current weather in the specified city.
    """

    city = city.strip().title()

    weather_data = {
        "Tokyo": {
            "condition": "Cloudy",
            "temperature_c": 15
        },
        "Osaka": {
            "condition": "Sunny",
            "temperature_c": 20
        },
        "Taipei": {
            "condition": "Rainy",
            "temperature_c": 22
        }
    }

    data = weather_data.get(city, {"condition": "Unknown", "temperature_c": 0})

    weather_result = WeatherOutput(city=city, condition=data["condition"], temperature_c=data["temperature_c"])

    return weather_result.model_dump_json()
    # return WeatherOutput(city=city, condition=data["condition"], temperature_c=data["temperature_c"])

print("================== TOOL INFORMATION ================")
print("\nTool name:")
print(get_weather.name)

print("\nTool description:")
print(get_weather.description)

print("\nTool input schema:")
schema_get_weather = get_weather.get_input_schema().model_json_schema()
print(json.dumps(schema_get_weather, ensure_ascii=False, indent=4))

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
    You are a helpful weather assistant.

    Use the get_weather tool when the user asks about weather.

    After receiving the tool result, return the final answer
    using the WeatherOutput format.

    """,
    response_format=WeatherOutput
)

result = agent.invoke({"messages": [{"role":"user", "content":"What's the weather in Tokyo?"}]})

result_json = "result.json"

pprint(result)

print("\n=========== MESSAGE FLOW ===============")

for index, message in enumerate(result["messages"], start=1):
    print(f"\nMessage {index}")
    print("Type:", type(message).__name__)

    if message.content:
        print("Content:")
        print(message.content)

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")

        for tool_call in tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])
            print("Tool call ID:", tool_call["id"])

structured_response = result.get("structured_response")

if structured_response:
    print("Type:")
    print(type(structured_response).__name__)

    print("\nPydantic object:")
    print(structured_response)

    print("\nJSON:")
    print(structured_response.model_dump_json(indent=4))
else:
    print("No structured response was returned.")

print("\n=============== FINAL ANSWER ==============")
if structured_response:
    print(f"The weather in {structured_response.city} is {structured_response.condition.lower()} with a temperature of {structured_response.temperature_c}°C.")
else:
    print(result["messages"][-1].content)



    