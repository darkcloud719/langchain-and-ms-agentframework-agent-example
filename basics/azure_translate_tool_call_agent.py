import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

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

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

def get_translation_guidelines(style: str) -> dict:
    """Based on translation style, return the corresponding guidelines."""

    guidelines_by_style = {
        "natural": "Translate naturally, using clear and commonly used English",
        "literal": "Translate closely to the original wording while keeping the English grammatical.",
        "formal": "Translate in a polite and formal style"
    }

    return {
        "style": style,
        "guideline": guidelines_by_style.get(style, guidelines_by_style["natural"])
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_translation_guidelines",
            "description": "Get translation guidelines based on the specified style",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["natural", "literal", "formal"],
                        "description": "The desired translation style"
                    }
                },
                "required": ["style"]
            }
        },
    }
]

messages = [
    {
        "role": "system",
        "content": "You translate Mandarin Chinese into English. Before translating, call get_translation_guidelines. Use the returned guidelines to produce the translation. Return only the translated sentence."
    },
    {
        "role": "user",
        "content": "Translate the following sentence into English using the natural style: 你好，世界。"
    }
]

response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages,
    tools=tools,
    tool_choice={
        "type": "function",
        "function": {
            "name": "get_translation_guidelines"
        }
    }
)

assistant_message = response.choices[0].message

messages.append(assistant_message.model_dump(exclude_none=True))

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)

        print(f"Function called: {function_name}")
        print(f"Arguments: {function_arguments}")

        if function_name == "get_translation_guidelines":
            function_result = get_translation_guidelines(style=function_arguments["style"])
        else:
            raise ValueError(f"Unknown function called: {function_name}")

        messages.append({
            "role":"tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_result, ensure_ascii=False)
        })

final_response = client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    messages=messages
)

print("Final response:")
print(final_response.choices[0].message.content)