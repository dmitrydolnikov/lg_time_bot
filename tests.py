from langchain_core.messages import HumanMessage

from main import get_utc_time, app
from datetime import datetime


def test_get_current_time_format():
    result = get_utc_time()
    assert "utc" in result, "Result should contain 'utc' key"

    # parse the datetime to ensure format is correct
    try:
        dt = datetime.fromisoformat(result["utc"].replace("Z", ""))
    except ValueError:
        assert False, f"- Invalid ISO format: {result['utc']}"

    print(f"+ test_get_current_time_format passed, time is {result['utc']}")

def test_user_messages():
    # Simulate user messages
    user_messages = [
        "Can you tell me the current time?",
        "What time is it in Cupertino?",
        "What is the current UTC time?",
        "Time check for Malvern, PA, please.",
        "Just checking the time for Cupertino.",
    ]

    for message in user_messages:
        state = {
            "messages": [HumanMessage(content=message)]
        }
        result = app.invoke(state)
        assert isinstance(message, str), f"- Invalid message type: {message}"
        print(f"+user message: {result['messages'][0].content}, result: {result['messages'][-1].content}")
        #print(f"full state: {result}")


if __name__ == "__main__":
    test_get_current_time_format()
    test_user_messages()