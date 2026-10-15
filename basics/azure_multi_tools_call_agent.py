import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
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

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

def get_translation_guidelines(style: str) -> dict:
    """Based on translation style, return the corresponding translation guidelines."""

    guidelines_by_style = {
        "natural": "Translate naturally, using clear and commonly used English",
        "literal": "Translate closely to the original wording while keeping the English grammatical",
        "formal": "Translate in a polite and formal style."
    }

    return {
        "style": style,
        "guideline": guidelines_by_style.get(style, guidelines_by_style["natural"])
    }

def get_weather(city: str) -> str:
    """Return the current weather for the specified city."""

    city_for_weather = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Windy, 18°C"
    }

    return city_for_weather.get(city, "Weather data not available")

def calculate(operation: str, number1: float, number2: float):
    """Perform a basic arithmetic operation on two numbers."""

    if operation == "add":
        return number1 + number2
    elif operation == "subtract":
        return number1 - number2
    elif operation == "multiply":
        return number1 * number2
    elif operation == "divide":
        if number2 == 0:
            raise ValueError("Cannot divide by zero")
        return number1 / number2
    else:
        raise ValueError(f"Unsupported operation: {operation}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_translation_guidelines",
            "description": "Get translation guidelines based on the specified style",
            "parameters": {
                "type":"object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["natural", "literal", "formal"],
                        "description": "The desired translation style"
                    }
                },
                "required": ["style"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for the specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city for which to get the weather"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
                    },
                    "number1": {
                        "type": "number",
                        "description": "The first number for the calculation"
                    },
                    "number2": {
                        "type": "number",
                        "description": "The second number for the calculation"
                    }
                },
                "required": ["operation", "number1", "number2"]
            }
        }
    }
]

messages = [
    {
        "role": "system", "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Translate the following Mandarin Chinese sentence into English: '天青色等煙雨，而我在等妳' in natural style"
        # "content": "What is the current weather in New York?"
    }
]

response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages,
    tools=tools
)

assistant_message = response.choices[0].message

messages.append(assistant_message.model_dump(exclude_none=True))

pprint(messages)

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)

        print(f"Function called: {function_name} with arguments: {function_arguments}")

        if function_name == "get_translation_guidelines":
            function_result = get_translation_guidelines(style=function_arguments["style"])
        elif function_name == "get_weather":
            function_result = get_weather(city=function_arguments["city"])
        elif function_name == "calculate":
            function_result = calculate(
                operation=function_arguments["operation"],
                number1=function_arguments["number1"],
                number2=function_arguments["number2"]
            )

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_result, ensure_ascii=False)
        })

final_response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages
)

print(f"Final response: {final_response.choices[0].message.content}")


