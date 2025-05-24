# lg_time_bot
simple time bot demo with langgraph
# Usage
git clone ```https://github.com/dmitrydolnikov/lg_time_bot.git```

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

langgraph dev

Create a file called `alibaba.key` and paste your Qwen-compatible API key in it.

# Langgraph queries
when you run langgraph dev, browser will open and show the query editor with graph state
you can run queries like this:
```json 
{
  "messages": [
    {
      "role": "user",
      "content": "What time is it?"
    }
  ]
}
```
or for a specific city:
```json 
{
  "messages": [
    {
      "role": "user",
      "content": "What time is it in New York?"
    }
  ]
}
```