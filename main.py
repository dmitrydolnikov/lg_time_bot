import os
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

import tzlocal
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import  ToolNode
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
def get_current_time(timezone:str="Etc/UTC") -> str:
    """Get the current time in the specified timezone.
    Timezone must be in IANA format (e.g. "UTC", "Europe/Berlin", "America/New_York").
    Examples:
    - "What time is it?" - timezone of the system if known, otherwise use UTC
    - "What time is it in Tokyo?" - timezone="Asia/Tokyo"
    - "Tell me the time in LA, Moscow and Budapest " - timezone="America/Los_Angeles", "Europe/Budapest", "Europe/Moscow"
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

tools = [get_current_time]

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
        messages = state.get("messages", [])

        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and last.content and not getattr(last, "tool_calls", None):
                print("Already finalized. Skipping.")
                return {"messages": messages+[], "message_type": "final"}

        if not messages:
            return {"messages": [AIMessage(content="Hi! Ask me the time.")], "message_type": "final"}

        llm_input = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(llm_input)

        if hasattr(response, "tool_calls") and response.tool_calls:
            return {"messages": messages + [response], "message_type": "tool_call"}

        if response.content.strip():
            return {"messages": messages + [response], "message_type": "final"}

        return {"messages": messages+[], "message_type": "final"}

    except Exception as e:
        print(f"agent_step error: {e}")
        raise


agent_node = RunnableLambda(agent_step)

# Graph structure
graph = StateGraph(State)

# Add agent node
graph.add_node("chatbot", agent_node)

# Add tool node using ToolNode
tool_node = ToolNode([get_current_time])
graph.add_node("get_current_time", tool_node)

#graph.add_edge(START, "chatbot")
# Entry point
graph.set_entry_point("chatbot") #this is sufficient to start the graph

#for conditional routing
def get_tool_name(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return last.tool_calls[0]["name"]
    if isinstance(last, ToolMessage):
        return "chatbot"
    return END



# Route to tool if requested
graph.add_conditional_edges("chatbot", get_tool_name)
#graph.add_edge("chatbot", "get_current_time")  #explicit name for Studio recognition

graph.add_edge("get_current_time", "chatbot")
#graph.add_edge("chatbot", END) # not needed, as the graph will finish when the chatbot node is done using finish point and message type = "final"
if False:
    graph.add_edge("chatbot", "get_current_time")  # for Studio only
# Compile the app
graph.set_finish_point("chatbot")
app = graph.compile()