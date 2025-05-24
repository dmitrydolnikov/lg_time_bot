#raise RuntimeError("🔥 If you see this in terminal, you are loading the correct file.")
import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("QWEN_API_KEY")
llm = ChatOpenAI(
    model="qwen-max",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    api_key=key
)


class State(TypedDict):
    messages: List[Dict[str, Any]]


def minimal_agent_step(state: dict) -> dict:
    print("📥 minimal_agent_step input state:", state)

    while isinstance(state, dict) and "values" in state:
        state = state["values"]

    raw_messages = state.get("messages", [])

    last_user_message = next((m["content"] for m in reversed(raw_messages) if m.get("type") == "human"), "Hello")
    response = f"You said: {last_user_message}"

    response_dict = {
        "type": "ai",
        "content": response
    }

    return {"messages": raw_messages + [response_dict]}


graph = StateGraph(State)
graph.add_node("minimal_agent", minimal_agent_step)
graph.set_entry_point("minimal_agent")
graph.set_finish_point("minimal_agent")
app1 = graph.compile()
