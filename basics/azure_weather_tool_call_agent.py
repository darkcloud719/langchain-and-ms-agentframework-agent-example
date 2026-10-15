import os, json
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
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def get_weather(city: str):

    city_for_weather = {
        "New York": "sunny 15 °C",
        "Los Angeles": "cloudy 20 °C",
        "Chicago": "rainy 10 °C"
    }
    return city_for_weather.get(city, "Weather data not available")

tools = []

tool_weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specified city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get the weather for."
                }
            }
        },
        "required": ["city"]
    }
}

tools.append(tool_weather)

messages = [
        {"role":"system", "content": "You are a helpful assistant."},
        {"role":"user", "content": "What's the weather like in New York?"}
    ]

response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages,
    tools=tools
)

assistant_message = response.choices[0].message
pprint(assistant_message)

messages.append(assistant_message.model_dump(exclude_none=True))

pprint(messages)

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)

        print(f"Function Called: {function_name} with arguments: {function_arguments}")

        if function_name == "get_weather":
            city = function_arguments.get("city")
            function_result = get_weather(city)
            print(f"Weather in {city}: {function_result}")

        else:
            raise ValueError(f"Unknown function called: {function_name}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_result, ensure_ascii=False)
        })

pprint(f"Final messages: {messages}")

final_response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages,
)

pprint(f"Final response: {final_response.choices[0].message.content}")




