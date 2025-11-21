import asyncio
import uuid
from pathlib import Path

from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.runners import InMemoryRunner
from IPython.display import display, Image as IPImage
import base64

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

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)
print("✅ MCP Tool created")

# Create image agent with MCP integration
image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)



runner = InMemoryRunner(agent=image_agent)
# Test the currency agent
async def main():
    response = await runner.run_debug("Provide a sample tiny image", verbose=True)

    # create dir for images
    img_dir = Path("images")
    img_dir.mkdir(exist_ok=True)

    img_count = 0
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if item.get("type") == "image":
                            img_data = base64.b64decode(item["data"])
                            img_count += 1
                            img_path = img_dir / f"tiny_image_{img_count}.png"
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                            print(f"✅ Image saved as {img_path}")

    if img_count == 0:
        print("⚠️ No images found in the response.")

if __name__ == "__main__":
    asyncio.run(main())


