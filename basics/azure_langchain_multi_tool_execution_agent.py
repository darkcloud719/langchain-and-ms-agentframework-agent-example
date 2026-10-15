import os
from dotenv import load_dotenv
from rich import print as pprint
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


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
def get_weather(city: str) -> str:
    """
    Get the current weather for a specified city.
    """

    weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C",
        "San Francisco": "Foggy, 15°C"
    }

    return weather_data.get(city, "Weather data not available for this city.")

@tool
def calculate(operation: str, number1: float, number2: float) -> float:
    """
    Perform a basic mathematical calculation. Supported operations: add, subtract, multiply, divide.
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
    Translate the given text to English.

    This demo returns a predifined translation
    for educational purposes.
    """

    translation_data = {
            "今天天氣很好":
                "The weather is nice today.",
    
            "我今天很開心":
                "I am very happy today."
    }

    return translation_data.get(text, "Translation not available for this text.")

tools = [get_weather, calculate, translate_to_english]

tools_by_name = {
    tool.name: tool for tool in tools
}

print(tools_by_name)

question = "What is the weather like in New York?"

llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage(content=question)]

response = llm_with_tools.invoke(messages)

pprint(response.tool_calls)


messages.append(response)

if response.tool_calls:
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        tool_call_id = tool_call["id"]

        pprint(f"\nExcecute Tool: {tool_name}")

        pprint(tool_args)

        selected_tool = tools_by_name[tool_name]

        tool_result = selected_tool.invoke(tool_args)

        pprint(f"Tool Result: {tool_result}")

        messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
        )

final_response = llm_with_tools.invoke(messages)

print(f"\n Final Answer : {final_response.content}")


    


