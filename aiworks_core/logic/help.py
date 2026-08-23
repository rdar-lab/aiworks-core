import logging
from typing import Any, Dict, List

from ..utils import async_to_sync
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from .llm import invoke_llm, get_prompts

logger = logging.getLogger(__name__)


def help_chat_reply(messages: List[Dict[str, Any]], manual_content: str) -> str:
    """Generate a chatbot reply for the help chat feature.

    Args:
        messages: Conversation history — list of ``{"role": "user"|"assistant", "content": ...}``
        manual_content: The user-manual text supplied by the frontend.

    Returns:
        The assistant's reply as a plain string.
    """
    prompts = get_prompts()
    system_prompt = prompts["help_chat"]["system_message"].format(
        manual_content=manual_content
    )

    lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return async_to_sync(invoke_llm)(
        "help_chat",
        messages=lc_messages,
        parse_json=False,
    )
