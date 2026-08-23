import logging

from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

logger = logging.getLogger(__name__)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def help_chat(request):
    """Help chatbot endpoint — answers questions about the app using the user manual.

    Always uses the fast LLM. Conversation history is passed in full on every
    call; nothing is persisted server-side.  The frontend is responsible for
    fetching and supplying ``manual_content`` with each request.
    """
    from ..logic.help import help_chat_reply

    messages = request.data.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return Response(
            {"error": "messages is required and must be a non-empty list"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    manual_content = request.data.get("manual_content", "")
    if not manual_content:
        return Response(
            {
                "error": "manual_content is required — the user manual must be available for the chatbot to function"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        reply = help_chat_reply(messages, manual_content)
        return Response({"reply": reply})
    except Exception as error:
        logger.exception("help_chat | error: %s", error)
        return Response(
            {"error": "An internal error occurred. Please try again."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
