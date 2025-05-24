import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from typing import Annotated
from zoneinfo import ZoneInfo

import tzlocal
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolCall, convert_to_messages
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import Annotated
from dotenv import load_dotenv
import langgraph
print(langgraph.__file__)
print("main.py loaded")


load_dotenv()
key = os.getenv("QWEN_API_KEY")
if not key:
    raise ValueError("Put QWEN_API_KEY=sk-xxxxxxxxxx into .env file")
    import sys
    sys.exit(1)

import re

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
    - "What time is it?" - timezone of the system if known, otherwise use UTC
    - "What time is it in Tokyo?" - timezone="Asia/Tokyo"
    - "Tell me the time in LA" - timezone="America/Los_Angeles"
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

# Define the system time zone from system settings
system_time_zone = tzlocal.get_localzone_name()  # This should be set based on the system's timezone

system_prompt = "You are an OpenAI-compatible assistant that uses tools through function calling. "\
                    "You MUST wait for the result of each tool call before replying. "\
                    "Tool outputs appear as messages of type 'tool' with a matching tool_call_id. "\
                    "When you see a tool result, explain it clearly to the user."\
                    "you are working in the timezone of the system, which is " + system_time_zone + ". "

# Wrap chatbot logic into a node
def agent_step(state: dict) -> dict:
    try:
        print("📥 Raw state at chatbot input:", state)

        # Normalize the incoming state from Studio
        while isinstance(state, dict) and "values" in state:
            state = state["values"]

        raw_messages = state.get("messages", [])

        # Safely normalize messages to BaseMessage objects
        clean_messages = []
        for m in raw_messages:
            if isinstance(m, dict):
                role = m.get("role") or m.get("type")
                content = m.get("content", "")
                if role in ("human", "user"):
                    clean_messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    clean_messages.append(AIMessage(content=content))
                elif role == "system":
                    clean_messages.append(SystemMessage(content=content))
                elif role == "tool":
                    tool_call_id = m.get("additional_kwargs", {}).get("tool_call_id", "") or m.get("tool_call_id", "")
                    clean_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                else:
                    print(f"⚠️ Skipped unknown message role/type: {role}")
            elif hasattr(m, "type"):  # already BaseMessage
                clean_messages.append(m)
            else:
                print(f"⚠️ Skipped unknown message format: {m}")

        # Check for list nesting issues
        if len(clean_messages) == 1 and isinstance(clean_messages[0], list):
            clean_messages = clean_messages[0]

        # Insert system prompt explicitly
        clean_messages.insert(0, SystemMessage(content=system_prompt))

        # Final explicit validation
        from langchain_core.messages import BaseMessage
        for i, msg in enumerate(clean_messages):
            assert isinstance(msg, BaseMessage), f"Non-BaseMessage at index {i}: {msg}"
        print("✅ clean_messages after validation:", clean_messages)

        # Now safely invoke the LLM
        response = llm_with_tools.invoke(clean_messages)
        print("🤖 agent_node response:", response)

        # Handle potential tool call
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_call_obj = ToolCall(
                name=tool_call["name"],
                args=tool_call["args"],
                id=tool_call["id"]
            )

            ai_msg = AIMessage(content="", tool_calls=[tool_call_obj])
            return {"messages": raw_messages + [ai_msg], "message_type": "tool_call"}


        # Always explicitly convert response back to dict for Studio
        response_dict = {
            "type": "ai",
            "content": response.content
        }
        final_messages = raw_messages + [response_dict]

        return {"messages": final_messages, "message_type": "final"}

    except Exception as e:
        print(f"❌ agent_step error: {e}")
        print(f"❌ final state at error: {state}")
        raise

agent_node = RunnableLambda(agent_step)

# Graph structure
graph = StateGraph(State)

# Add agent node
graph.add_node("chatbot", RunnableLambda(agent_step))

# Add tool node using ToolNode
tool_node = ToolNode([get_current_time_tool])
graph.add_node("get_current_time_tool", tool_node)

#graph.add_edge(START, "chatbot")
# Entry point
graph.set_entry_point("chatbot") #this is sufficient to start the graph

#for confitional routing
tool_to_node_map = {
    "get_current_time": "get_current_time_tool",
    "get_current_time_tool": "get_current_time_tool"
}

def get_tool_name(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        tool_name = last.tool_calls[0]["name"]
        return tool_to_node_map.get(tool_name, "END")
    return "END"

# Route to tool if requested
graph.add_conditional_edges("chatbot", get_tool_name)

graph.add_edge("get_current_time_tool", "chatbot")

# Compile the app
graph.set_finish_point("chatbot")
app = graph.compile()