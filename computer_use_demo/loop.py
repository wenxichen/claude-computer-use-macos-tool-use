"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

import json
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from openai import OpenAI

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
    

######
# RAG logging
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
######

# BETA_FLAG = "computer-use-2024-10-22"


# class APIProvider(StrEnum):
#     ANTHROPIC = "anthropic"
#     BEDROCK = "bedrock"
#     VERTEX = "vertex"

# PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
#     APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
#     APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
#     APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
# }

# if provider == APIProvider.ANTHROPIC:
#     client = Anthropic(api_key=api_key)
# elif provider == APIProvider.VERTEX:
#     client = AnthropicVertex()
# elif provider == APIProvider.BEDROCK:
#     client = AnthropicBedrock()

# ================================
# TODO: Support chatbot query and update.
# 1. check if chatbot is enabled
# 2. if chatbot is enabled, query the chatbot for information about the task
# 3. if chatbot returns information, check if further query is needed
# 4. if further query is needed, query for further information
# 5. use the information to generate the plan
# 6. carry out task following the plan
# 7. if plan is not successful or need human interaction, update chatbot with new information
# The idea is the chatbot can be automatically updated from user actions or interaction,
# use only simple flow and Q&A for now
# ================================



# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
# <SYSTEM_CAPABILITY>
# * You can use the bash tool to execute commands in the terminal.
# * To open applications, you can use the `open` command in the bash tool. For example, `open -a Safari` to open the Safari browser.
# * When using your bash tool with commands that are expected to output very large quantities of text, redirect the output into a temporary file and use `str_replace_editor` or `grep -n -B <lines before> -A <lines after> <query> <filename>` to inspect the output.
# * When using your computer function calls, they may take a while to run and send back to you. Where possible and feasible, try to chain multiple of these calls into one function call request.
# <IMPORTANT>
# * If the item you are looking at is a PDF, and after taking a single screenshot of the PDF it seems you want to read the entire document, instead of trying to continue to read the PDF from your screenshots and navigation, determine the URL, use `curl` to download the PDF, install and use `pdftotext` (you may need to install it via `brew install poppler`) to convert it to a text file, and then read that text file directly with your `str_replace_editor` tool.
WORKER_SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are a worker agent utilizing a MacOS computer using {platform.machine()} architecture with internet access.
* You have a manager that may provide you with plan and suggestions for how to complete the task.
* You and your manager will have access to the screen of the computer.
* When viewing a page, it can be helpful to zoom out so that you can see everything on the page. Alternatively, ensure you scroll down to see everything before deciding something isn't available.
* When instruction is provided to you through the text editor, please read it carefully, and follow it along with the manager's plan to complete the task.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Safari or other applications, if any startup wizards or prompts appear, **IGNORE THEM**. Do not interact with them. Instead, click on the address bar or the area where you can enter commands or URLs, and proceed with your task.
</IMPORTANT>
"""

MANAGER_SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are a manager of two agents: one worker agent that utilizes a MacOS computer using {platform.machine()} architecture with internet access, and one quality assurance agent that will review the worker agent's output.
* You and the agents will have access to the screen of the computer.
* You can evaluate the goal and the progress from the agents to decide what the agents should do next.
* When you are not sure what the agent should do next, you can ask the agent to search for relevent information on Google using the firefox browser.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* Please do not use any tools! Just provide a plan in text format for the worker agent to complete the task.
</IMPORTANT>
"""

QA_SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are a quality assurance agent.
* You are working with a worker agent and a manager. All three of you will have access to the screen of the computer.
* You will review the worker agent's output and determine if the task defined by the original instruction is completed.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
* Please output in JSON format with the following keys:
    * `is_complete`: boolean indicating if the worker agent's output is correct and complete and meets the goal defined by the original instruction.
    * `feedback`: string providing feedback on the worker agent's output.
* Here is an example JSON output::
    {{"is_complete": true, "feedback": "The worker agent's output is correct and complete and meets the goal defined by the original instruction."}}
</SYSTEM_CAPABILITY>

<IMPORTANT>
* Please do not use any tools! Just respond with a JSON output.
</IMPORTANT>
"""

def _print_chatbot_messages(messages: list[str], role: str):
    for message in messages:
        print(f"{role}:", message)

def _store_chatbot_messages(all_chatbot_messages: list[str], messages: list[str], role: str):
    for message in messages:
        all_chatbot_messages.append((role, message))

def _user_message_to_check_further(instruction: str, all_chatbot_messages: list[str]):
    chatbot_messages = "\n".join([f"{role}: {message}" for role, message in all_chatbot_messages])
    return (f"Given the INSTRUCTION and response from Juji chatbot below, please advise if further query is needed from the chatbot."
            f"\n\nINSTRUCTION: {instruction}"
            f"\n\nquery history: \"\"\"\n{chatbot_messages}\n\"\"\""
            f"\n\nPlease respond in JSON format with the following keys: "
            f"\n\t- \"further query needed\": boolean indicating if further query is needed"
            f"\n\t- \"chatbot does not know\": boolean indicating if the chatbot does not know the answer"
            f"\n\t- \"query suggestion\": string providing a suggestion for a query to Juji"
            f"\n\nHere are some examples of JSON output:"
            f"\n\t{{\"further query needed\": true, \"chatbot does not know\": false, \"query suggestion\": \"What is the capital of France?\"}}"
            f"\n\t{{\"further query needed\": false, \"chatbot does not know\": true, \"query suggestion\": \"\"}}"
            f"\n\t{{\"further query needed\": false, \"chatbot does not know\": false, \"query suggestion\": \"\"}}")

def _query_chatbot(chatbot: Chatbot, instruction: str, text_query_client: OpenAI):
    """Query the chatbot for information about the task"""
    all_chatbot_messages = []
    further_query_count = 0

    print("Querying Juji for info about: ", instruction)
    participation = chatbot.start_chat()
    juji_messages = participation.get_messages()
    _print_chatbot_messages(juji_messages, "Juji")
    _store_chatbot_messages(all_chatbot_messages, juji_messages, "Juji")

    send_to_juji = "Do you know anything about: " + instruction
    print("You:", send_to_juji)
    _store_chatbot_messages(all_chatbot_messages, [send_to_juji], "You")
    juji_messages = participation.send_chat_msg(send_to_juji)
    _print_chatbot_messages(juji_messages, "Juji")
    _store_chatbot_messages(all_chatbot_messages, juji_messages, "Juji")

    # check if further query is needed
    user_message = _user_message_to_check_further(instruction, all_chatbot_messages)
    response = text_query_client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": user_message}],
    )
    print("Text query response:", response)
    response_json = json.loads(response.choices[0].message.content)

    while further_query_count < 3 and response_json.get("further query needed", False) and not response_json.get("chatbot does not know", False) and response_json.get("query suggestion", ""):
        print("You:", response_json.get("query suggestion"))
        further_query_count += 1
        _store_chatbot_messages(all_chatbot_messages, [response_json.get("query suggestion")], "You")
        juji_messages = participation.send_chat_msg(response_json.get("query suggestion"))
        _print_chatbot_messages(juji_messages, "Juji")
        _store_chatbot_messages(all_chatbot_messages, juji_messages, "Juji")
        user_message = _user_message_to_check_further(instruction, all_chatbot_messages)
        response = text_query_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": user_message}],
        )
        print("Text query response:", response)
        response_json = json.loads(response.choices[0].message.content)
    
    print("End of intial chatbot query")

    return all_chatbot_messages



def _manager_check_progress(
        messages: list[BetaMessageParam], 
        computer_use_client: Anthropic, 
        model: str, 
        manager_system: str, 
        api_response_callback: Callable[[APIResponse[BetaMessage]], None],
        tool_collection: ToolCollection,
        session_number: int,
        all_chatbot_messages: list[str]
):
    


    if session_number > 0:
        user_message = {
            "role": "user",
            "content": 
                f"Given the INSTRUCTION, previous steps, and the previous plan, "
                "please adjust the plan for the agent to continue completing the task. Please do not use any tools. "
                "If the worker agent is stuck, you can ask the worker agent to search for relevent information on Google.",
        }
    else:

        # if Juji returns info, use it to generate the plan
        if all_chatbot_messages:
            initial_chatbot_query_messages = "\n".join([f"{role}: {message}" for role, message in all_chatbot_messages])
            user_message = {
                "role": "user",
                "content": 
                    (f"Given the INSTRUCTION and context, as well as the following interaction history between you and Juji, "
                    f"\n\ninteraction history: \"\"\"\n{initial_chatbot_query_messages}\n\"\"\""
                    f"please provide a plan for the agent to complete the task. Please do not use any tools."
                    )
            }
        else:
            user_message = {
                "role": "user",
                "content": 
                    f"Given the INSTRUCTION and context, "
                    "please provide a plan for the agent to complete the task. Please do not use any tools.",
            }
    
    # Call the API to get some planning and context
    print("User message:", user_message)
    raw_response = computer_use_client.beta.messages.with_raw_response.create(
        max_tokens=1024,
        system=manager_system,
        messages=messages + [user_message],
        tools=tool_collection.to_params(),
        model=model,
        betas=["computer-use-2024-10-22"]
    )

    api_response_callback(cast(APIResponse[BetaMessage], raw_response), role="manager", session_number=session_number) 

    response = raw_response.parse()

    return response.content[0].text

def _manager_report_progress(
    messages: list[BetaMessageParam], 
    computer_use_client: Anthropic, 
    model: str, 
    manager_system: str,
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    tool_collection: ToolCollection,
):
    
    # Call the API to get some planning and context
    raw_response = computer_use_client.beta.messages.with_raw_response.create(
        max_tokens=1024,
        system=manager_system,
        messages=messages + [{
            "role": "user",
            "content": 
                f"Given the INSTRUCTION, what the worker agent has done, and the QA agent's assessment (if any), "
                "please generate a short report on what has been done and whether the goal has been achieved.",
         }],
        tools=tool_collection.to_params(),
        model=model,
        betas=["computer-use-2024-10-22"]
    )

    response = raw_response.parse()
    api_response_callback(None, role="manager", final_report=response.content[0].text) 

async def sampling_loop(
    *,
    model: str,
    computer_use_client: Anthropic,
    text_query_client: OpenAI,
    messages: list[BetaMessageParam],
    instruction: str,
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    rag_url: str | None = None,
    chatbot_link: str | None = None,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )

    system = (
        f"{WORKER_SYSTEM_PROMPT}\n\n<INSTRUCTION>\n{instruction}\n</INSTRUCTION>"
    )

    manager_system = (
        f"{MANAGER_SYSTEM_PROMPT}\n\n<INSTRUCTION>\n{instruction}\n</INSTRUCTION>"
    )

    qa_system = (
        f"{QA_SYSTEM_PROMPT}\n\n<INSTRUCTION>\n{instruction}\n</INSTRUCTION>"
    )
    # Overwrite the messages with the manager's plan
    # messages=[]

    if rag_url:
        documents = SimpleWebPageReader(html_to_text=True).load_data([rag_url])
        index = SummaryIndex.from_documents(documents)
        retriever = index.as_retriever()
        nodes = retriever.retrieve(instruction)
        relevent_context = "\n".join([node.get_content() for node in nodes] )
        messages.append(
            {
                "role": "user",
                "content": f"Here is some relevent context for the task:\n{relevent_context}"
            }
        )
        print(relevent_context)

    all_chatbot_messages = []
    if chatbot_link:
        from juji_python_sdk import Chatbot
        chatbot = Chatbot(chatbot_link)
        all_chatbot_messages = _query_chatbot(chatbot, instruction, text_query_client)

    total_sessions = 0

    while total_sessions < 10:

        manager_plan = _manager_check_progress(messages, computer_use_client, model, manager_system, api_response_callback, tool_collection, session_number=total_sessions, all_chatbot_messages=all_chatbot_messages)

        if total_sessions == 0:
            messages.append(
                {
                    "role": "user",
                    "content": f"Given the INSTRUCTION, here is a plan provided by the manager:\n{manager_plan}"
                    "\n\nPlease follow the plan to complete the task.",
                }
            )

        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Given the INSTRUCTION and what you have done so far, here is an updated plan provided by the manager:\n{manager_plan}"
                    "\n\nPlease follow the plan to complete the task.",
                }
            )

        count = 0

        while count < 5:
            if only_n_most_recent_images:
                _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

            # Call the API
            # we use raw_response to provide debug information to streamlit. Your
            # implementation may be able call the SDK directly with:
            # `response = client.messages.create(...)` instead.
            raw_response = computer_use_client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=system,
                tools=tool_collection.to_params(),
                betas=["computer-use-2024-10-22"],
            )

            api_response_callback(cast(APIResponse[BetaMessage], raw_response), count, session_number=total_sessions)

            response = raw_response.parse()

            messages.append(
                {
                    "role": "assistant",
                    "content": cast(list[BetaContentBlockParam], response.content),
                }
            )

            tool_result_content: list[BetaToolResultBlockParam] = []
            for content_block in cast(list[BetaContentBlock], response.content):
                output_callback(content_block)
                if content_block.type == "tool_use":
                    result = await tool_collection.run(
                        name=content_block.name,
                        tool_input=cast(dict[str, Any], content_block.input),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block.id)
                    )
                    tool_output_callback(result, content_block.id)

            if not tool_result_content:
                # Check with QA agent if goal is met
                qa_response = computer_use_client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages + [{
                        "role": "user", 
                        "content": f"Has the instruction goal been achieved? Please answer in JSON format."
                    }],
                    model=model,
                    system=qa_system,
                    tools=tool_collection.to_params(),
                    betas=["computer-use-2024-10-22"]
                )
            
                api_response_callback(cast(APIResponse[BetaMessage], qa_response), count, role="qa", session_number=total_sessions)
                qa_result = qa_response.parse()
                
                qa_json = json.loads(qa_result.content[0].text)
                if qa_json.get('is_complete', False):
                    api_response_callback(None, is_done=True)
                    messages.append({"content": qa_result.content[0].text, "role": "assistant"})  
                    _manager_report_progress(messages, computer_use_client, model, manager_system, api_response_callback, tool_collection)
                    return messages
            messages.append({"content": tool_result_content, "role": "user"})
        
            count += 1

        total_sessions += 1

    _manager_report_progress(messages, computer_use_client, model, manager_system, api_response_callback, tool_collection)


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
