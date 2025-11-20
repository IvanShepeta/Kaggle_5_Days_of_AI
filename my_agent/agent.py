from google.adk.agents.llm_agent import Agent
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# try:
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     print("✅ Gemini API key setup complete.")
# except Exception as e:
#     print(
#         f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
#     )

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

runner = InMemoryRunner(agent=root_agent)

async def main():
    response = await runner.run_debug("What's the weather in London?")
    print(response)


asyncio.run(main())