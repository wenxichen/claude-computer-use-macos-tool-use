import asyncio
import os
import sys
import json
import base64
import agentops

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage, BetaMessageParam
from anthropic import APIResponse

import signal
import pickle
from pathlib import Path

agentops.init(api_key="f03a808a-f077-4f37-9af0-f983707614de")

def save_messages(messages):
    """Save messages to a pickle file"""
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/messages.pkl", "wb") as f:
        pickle.dump(messages, f)
    print("\nSaved current progress to checkpoints/messages.pkl")

def load_messages():
    """Load messages from a pickle file if it exists"""
    try:
        with open("checkpoints/messages.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
def signal_handler(signum, frame):
    """Handle SIGINT by saving messages and exiting"""
    print("\nReceived interrupt signal. Saving progress...")
    if hasattr(signal_handler, 'messages'):
        save_messages(signal_handler.messages)
    sys.exit(0)

async def main():
    # Set up your Anthropic API key and model
    api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    if api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Please first set your API key in the ANTHROPIC_API_KEY environment variable"
        )
    provider = APIProvider.ANTHROPIC

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Check if the instruction is provided via command line arguments
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
        if len(sys.argv) > 2:
            rag_url = sys.argv[2]
        else:
            rag_url = None  
    else:
        instruction = "Save an image of a cat to the desktop."
        rag_url = None
    print(
        f"Starting Claude 'Computer Use'.\nPress ctrl+c to stop.\nInstructions provided: '{instruction}'"
    )

    # Set up the initial messages
    messages: list[BetaMessageParam] = [
  
    ]

    # Define callbacks (you can customize these)
    def output_callback(content_block):
        if isinstance(content_block, dict) and content_block.get("type") == "text":
            print("Assistant:", content_block.get("text"))

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        if result.output:
            print(f"> Tool Output [{tool_use_id}]:", result.output)
        if result.error:
            print(f"!!! Tool Error [{tool_use_id}]:", result.error)
        if result.base64_image:
            # Save the image to a file if needed
            os.makedirs("screenshots", exist_ok=True)
            image_data = result.base64_image
            with open(f"screenshots/screenshot_{tool_use_id}.png", "wb") as f:
                f.write(base64.b64decode(image_data))
            print(f"Took screenshot screenshot_{tool_use_id}.png")

    def api_response_callback(response: APIResponse[BetaMessage], step: int=None, role: str = "worker", is_done: bool = False, final_report: str = None, session_number: int = None):
        if is_done:
            print("\n---------------\nQA think it is Done")
            return
        else:
            if role == "manager":
                if final_report:
                    print(
                        "\n================\nManager",
                        "\nFinal Report:\n",
                        final_report,
                        "\n================\n",
                    )
                else:
                    print(
                        "\n---------------\nSession: ", session_number+1, " | Manager",
                        "\nAPI Response:\n",
                        json.dumps(json.loads(response.text)["content"], indent=4),  # type: ignore
                        "\n",
                    )
            elif role == "qa":
                print(
                    "\n---------------\nSession: ", session_number+1, " | QA",
                    "\nAPI Response:\n",
                    json.dumps(json.loads(response.text)["content"], indent=4),  # type: ignore
                    "\n",
                )
            elif role == "worker":
                print(
                    "\n---------------\nSession: ", session_number+1, " Step:",
                    step+1,
                    "\nAPI Response:\n",
                    json.dumps(json.loads(response.text)["content"], indent=4),  # type: ignore
                    "\n",
                )
            else:
                raise ValueError(f"Invalid role: {role}")
            
    # Check for saved messages
    saved_messages = load_messages()
    if saved_messages:
        print("Found saved progress. Would you like to continue? (y/n)")
        if input().lower() == 'y':
            messages = saved_messages
            print("Continuing from saved progress...")
        else:
            print("Starting fresh...")

    # Store messages in signal handler for access during interrupt
    signal_handler.messages = messages

    # Run the sampling loop
    messages = await sampling_loop(
        model="claude-3-5-sonnet-20241022",
        provider=provider,
        system_prompt_suffix="",
        messages=messages,
        instruction=instruction,
        rag_url=rag_url,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        api_response_callback=api_response_callback,
        api_key=api_key,
        only_n_most_recent_images=10,
        max_tokens=4096,
    )

    # Save final messages
    save_messages(messages)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Encountered Error:\n{e}")

    agentops.end_session('Success')
