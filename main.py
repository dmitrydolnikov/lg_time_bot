from datetime import datetime, timezone
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
import re


def get_current_time() -> dict:
    """Return the current UTC time in ISO‑8601 format.
    Example → {"utc": "2025‑05‑21T06:42:00Z"}"""
    current_time = datetime.now(timezone.utc).isoformat() + 'Z' # Append 'Z' to indicate UTC
    return {"utc": current_time}

# Logic node, no llm yet
def test_agent_node(state: dict) -> dict:

    if not state.get("messages"):
        print (f"warning: No messages in state")
        return {"messages": [AIMessage(content="Hi! Ask me what time it is.")]}

    last_msg = state["messages"][-1].content.lower()
    if re.search(r'\bwhat time\b|\bcurrent time\b|\btime is it\b', last_msg):
        result = get_current_time()
        reply = f"The current UTC time is {result['utc']}"
    else:
        reply = "I'm a simple bot. Ask me the time!"
    return {"messages": state["messages"] + [AIMessage(content=reply)]}

# Define state and graph
graph = StateGraph(dict)
graph.add_node("respond", test_agent_node)
graph.set_entry_point("respond")
graph.set_finish_point("respond")

app = graph.compile()
