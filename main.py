from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolCall
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import Annotated


import re

with open("alibaba.key", "r") as f:
    key = f.read().strip()

llm = ChatOpenAI(
    model="qwen-max",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    api_key=key
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def get_utc_time() -> dict:
    """Return the current UTC time in ISO‑8601 format.
    Example → {"utc": "2025‑05‑21T06:42:00Z"}"""
    current_time = datetime.now(timezone.utc).isoformat() + 'Z' # Append 'Z' to indicate UTC
    return {"utc": current_time}

# test Logic node, no llm yet
def test_agent_node(state: dict) -> dict:

    if not state.get("messages"):
        print (f"warning: No messages in state")
        return {"messages": [AIMessage(content="Hi! Ask me what time it is.")]}

    last_msg = state["messages"][-1].content.lower()
    if re.search(r'\bwhat time\b|\bcurrent time\b|\btime is it\b', last_msg):
        result = get_utc_time()
        reply = f"The current UTC time is {result['utc']}"
    else:
        reply = "I'm a simple bot. Ask me the time!"
    return {"messages": state["messages"] + [AIMessage(content=reply)]}

@tool
def get_current_time_tool(timezone:str="Etc/UTC") -> str:
    """Get the current time in the specified timezone.

    Timezone must be in IANA format (e.g. "UTC", "Europe/Berlin", "America/New_York").

    Examples:
    - "What time is it?" - use default timezone=Etc/UTC
    - "What time is it in Tokyo?" → timezone="Asia/Tokyo"
    - "Tell me the time in LA" → timezone="America/Los_Angeles"
    """
    #ZoneInfo default timezone is Etc/UTC, so if no timezone is provided, it will return UTC time
    if timezone.upper() == "UTC":
        timezone = "Etc/UTC"
    try:
        now = datetime.now(ZoneInfo(timezone)).isoformat()
        return f"The current {timezone} time is {now}"
    except Exception as e:
        return f"Error: {e}. Try without timezone, or use a valid timezone like 'UTC', 'America/New_York', or 'Asia/Tokyo'."

def chatbot_node(state: dict) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="Hi! I am a simple chatbot, can tell what is current time.")]}


    last = messages[-1]
    if isinstance(last, HumanMessage):
        content = last.content.lower()
        if "time" in content:
            return {"messages": messages, "next_tool": "get_current_time"}
        else:
            return {"messages": messages + [
                AIMessage(content="I can tell you the current time. Ask me about it")]}

    return {"messages": messages}

tools = [get_current_time_tool]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# Wrap chatbot logic into a node
def agent_step(state: dict) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are an OpenAI-compatible assistant that uses tools through function calling. "
                "You MUST wait for the result of each tool call before replying. "
                "Tool outputs appear as messages of type 'tool' with a matching tool_call_id. "
                "When you see a tool result, explain it clearly to the user."
            )
        ),
        *state["messages"]
    ]
    response = llm_with_tools.invoke(messages)
    print("agent_node: messages =", messages),
    print("agent_node: response =", response),
    # Check for OpenAI-style tool call
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_call = ToolCall(
            name=tool_call["name"],
            id=tool_call["id"],
            args=tool_call["args"],
        )
        return {
            "messages": state["messages"] + [AIMessage(content="", tool_calls=[tool_call])]
        }
    # If no tool call, return the response
    return {
        "messages": messages + [response]

    }

agent_node = RunnableLambda(agent_step)

# Graph structure
graph = StateGraph(State)

# Add agent node
graph.add_node("chatbot", RunnableLambda(agent_step))

# Add tool node using ToolNode
tool_node = ToolNode([get_current_time_tool])
graph.add_node("get_current_time_tool", tool_node)

# Entry point
graph.set_entry_point("chatbot")

# Route to tool if requested
graph.add_conditional_edges(
    "chatbot",
    lambda state: (
        state["messages"][-1].tool_calls[0]["name"]
        if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls
        else "END"
    )
)

# Loop tool response back to chatbot
graph.add_edge("get_current_time_tool", "chatbot")

# Compile the app
graph.set_finish_point("chatbot")
app = graph.compile()