from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


@tool
def remove_file(path: str) -> str:
    """Simulate deleting a file."""
    return f"[SIMULATION] Deleted {path}"


@tool
def fetch_file(path: str) -> str:
    """Simulate reading a file."""
    return f"[SIMULATION] Contents of {path}"


@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email."""
    return f"[SIMULATION] Sent email to {to}"


checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[remove_file, fetch_file, notify_email],
    interrupt_on={
        # 刪除前暫停；允許核准、修改參數或拒絕
        "remove_file": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },

        # 讀取檔案不會改變資料，不需要暫停
        "fetch_file": False,

        # 寄信前暫停；只允許核准或拒絕
        "notify_email": {
            "allowed_decisions": ["approve", "reject"]
        },
    },
    checkpointer=checkpointer,
)


# 暫停與恢復必須使用同一個 thread_id
config = {
    "configurable": {
        "thread_id": "hitl-practice-001"
    }
}


# 請模型提出需要使用工具的操作
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please delete the file /tmp/old_report.txt. "
                    "Before sending any email, ask me for approval."
                ),
            }
        ]
    },
    config=config,
)


# 顯示是否暫停，以及待審核的工具呼叫
interrupts = result.get("__interrupt__", [])

if not interrupts:
    print("Agent did not pause. Check the request and tool call.")
else:
    print("Approval required:")
    print(interrupts)

    decision = input(
        "\nApprove or reject? (a/r): "
    ).strip().lower()

    if decision == "a":
        resume_command = Command(
            resume={
                "decisions": [
                    {
                        "type": "approve"
                    }
                ]
            }
        )

    elif decision == "r":
        resume_command = Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": "The user rejected this action.",
                    }
                ]
            }
        )

    else:
        raise ValueError("Please enter 'a' to approve or 'r' to reject.")

    # 用相同的 config / thread_id 恢復流程
    final_result = agent.invoke(
        resume_command,
        config=config,
    )

    print("\nMessages after resume:")
    for message in final_result["messages"]:
        print(type(message).__name__, ":", message.content)