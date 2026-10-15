import os
from dotenv import load_dotenv
from rich import print as pprint
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
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
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

@tool
def get_weather(city:str) -> str:
    """
    Get the current weather for a specified city.
    """

    weather_data ={
        "New York": "Sunny, 25°C",
        "London": "Cloudy, 18°C",
        "Chicago": "Windy, 20°C",
        "Taipei": "Rainy, 22°C"
    }

    return weather_data.get(city, "Weather data not available.")

@tool
def calculate(
    operation: str,
    number1: float,
    number2: float
) -> float:
    """
    Perform a basic mathematical calculation.

    Supported operations:
    add, subtract, multiply, divide
    """

    if operation == "add":
        return number1 + number2
    elif operation == "subtract":
        return number1 - number2
    elif operation == "multiply":
        return number1 * number2
    elif operation == "divide":

        if number2 == 0:
            raise ValueError("Cannot divide by zero.")

        return number1 / number2
    else:
        raise ValueError(f"Unsupported operation: {operation}")

@tool
def translate_to_english(text: str) -> str:
    """
    Translate Mandarin Chinese text into English.

    This demo returns a predefined translation
    for educational purposes.
    """

    translation_data = {
        "今天天氣很好": "The weather is nice today.",
        "我今天很開心": "I am very happy today."
    }

    return translation_data.get(text, "Translation not found.")

tools = [get_weather, calculate, translate_to_english]

llm_with_tools = llm.bind_tools(tools)

question = "What is 25 multiplied by 8?"

response = llm_with_tools.invoke(question)

pprint(response)


if response.tool_calls:
    for tool_call in response.tool_calls:
        pprint(f"Tool name: {tool_call['name']}, Arguments: {tool_call['args']}, Id: {tool_call['id']}")
else:
    pprint("No tools were called.")