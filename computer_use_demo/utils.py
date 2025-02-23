import pickle
from pathlib import Path
import os
import base64
import json
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage
from anthropic import APIResponse


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
    
def remove_checkpoints():
    """Remove all checkpoints"""
    for file in Path("checkpoints").glob("*"):
        file.unlink()
    print("Checkpoints removed.")
    
# ================================
# Callbacks
# ================================
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