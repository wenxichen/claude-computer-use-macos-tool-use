import asyncio
import os
import sys
import dotenv
import signal

from anthropic import Anthropic
from openai import OpenAI
from computer_use_demo.loop import sampling_loop, _init_chatbot
from anthropic.types.beta import BetaMessageParam

from computer_use_demo.utils import save_messages, load_messages, remove_checkpoints, output_callback, tool_output_callback, api_response_callback

dotenv.load_dotenv()

from juji_python_sdk import Chatbot, Participation

# ================================
# Basic setup
# Set up your Anthropic API key and model
api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
if api_key == "YOUR_API_KEY_HERE":
    raise ValueError(
        "Please first set your API key in the ANTHROPIC_API_KEY environment variable or in the .env file."
    )
computer_use_client = Anthropic(api_key=api_key)

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
if openai_api_key == "YOUR_API_KEY_HERE":
    raise ValueError(
        "Please first set your API key in the OPENAI_API_KEY environment variable or in the .env file."
    )
text_query_client = OpenAI(api_key=openai_api_key)

# # Set up your AgentOps API key
# agentops_api_key = os.getenv("AGENTOPS_API_KEY", "YOUR_API_KEY_HERE")
# if agentops_api_key == "YOUR_API_KEY_HERE":
#     print(
#         "skipping agentops init as no key is set."
#     )
#     USE_AGENTOPS = False
# else:
#     import agentops
#     agentops.init(api_key=agentops_api_key)
#     USE_AGENTOPS = True

chatbot_link = os.getenv("CHATBOT_LINK")
if not chatbot_link:
    print(
        "skipping chatbot init as no link is set."
    )

juji_api_key = os.getenv("JUJI_API_KEY")
juji_platform_url = os.getenv("JUJI_PLATFORM_URL")
juji_chatbot_engagement_id = os.getenv("JUJI_CHATBOT_ENGAGEMENT_ID")
if not juji_api_key or not juji_chatbot_engagement_id:
    print(
        "Running without Juji chatbot update as no API key or engagement ID is provided."
    )


# def signal_handler(signum, frame):
#     """Handle SIGINT by saving messages and exiting"""
#     print("\nReceived interrupt signal. Saving progress...")
#     if hasattr(signal_handler, 'messages'):
#         save_messages(signal_handler.messages)
#     sys.exit(0)

# # Set up signal handler
# signal.signal(signal.SIGINT, signal_handler)

# Check if the instruction is provided via command line arguments
if len(sys.argv) > 1:
    instruction = sys.argv[1]
    if len(sys.argv) > 2:
        rag_url = sys.argv[2]
    else:
        rag_url = None  
else:
    instruction = "If a text editor is opened on the screen with tasks or steps to complete, please follow the instructions and complete the tasks using the broswer that is already opened. Otherwise, please do nothing."
    rag_url = None

# ================================


async def main(messages: list[BetaMessageParam] = None, human_intervention: bool = False, all_chatbot_messages: list[BetaMessageParam] = None, chatbot_participation: Participation = None):

    print(
        f"Starting Claude 'Computer Use'.\nPress ctrl+c to stop.\nInstructions provided: '{instruction}'"
    )

    if not messages:
        # Check for saved messages
        saved_messages = load_messages()
        if saved_messages:
            print("Found saved progress. Would you like to continue? (y/n)")
            if input().lower() == 'y':
                messages = saved_messages
                print("Continuing from saved progress...")
            else:
                print("Starting fresh...")
                remove_checkpoints()

    # # Store messages in signal handler for access during interrupt
    # signal_handler.messages = messages

    # Run the sampling loop
    messages = await sampling_loop(
        model="claude-3-5-sonnet-20241022",
        computer_use_client=computer_use_client,
        text_query_client=text_query_client,
        messages=messages,
        instruction=instruction,
        rag_url=rag_url,
        chatbot_link=chatbot_link,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        api_response_callback=api_response_callback,
        only_n_most_recent_images=10,
        max_tokens=4096,
        juji_api_key=juji_api_key,
        juji_chatbot_engagement_id=juji_chatbot_engagement_id,
        juji_platform_url=juji_platform_url,
        human_intervention=human_intervention,
        all_chatbot_messages=all_chatbot_messages,
        chatbot_participation=chatbot_participation
    )

    # Save final messages
    save_messages(messages)


if __name__ == "__main__":
    if chatbot_link:
        chatbot = Chatbot(chatbot_link)
        all_chatbot_messages, chatbot_participation = _init_chatbot(chatbot)
    else:
        all_chatbot_messages = None
        chatbot_participation = None
    messages = []
    human_intervention = False
    while True:
        try:
            asyncio.run(main(messages=messages, human_intervention=human_intervention, all_chatbot_messages=all_chatbot_messages, chatbot_participation=chatbot_participation))
            break
        except KeyboardInterrupt:
            user_input = input("At MAIN:\n- If you want to stop the program, press Ctrl+C again.\n- If you want to continue, press Enter.\n- If you want to intervene with additional instructions, press 'i'.")
            human_intervention = True
            if messages:
                if messages[-1]["role"] == "assistant" and messages[-1]["content"][-1].type == "tool_use":
                    print("popping last message since tool command not processed")
                    messages.pop()
            if user_input == "i":
                instruction = input("Please provide the additional instructions:")
                messages.append({"content": f"The human user intervened the agent with the following additional instructions: {instruction}\n\nplease adjust the plan for the agent to continue completing the task. Please do not use any tools.", "role": "user"})
            else:
                messages.append({"content": f"The human user intervened.\n\nplease adjust the plan for the agent to continue completing the task. Please do not use any tools.", "role": "user"})

    # except Exception as e:
    #     print(f"Encountered Error:\n{e}")

    # if USE_AGENTOPS:    
    #     agentops.end_session('Success')
