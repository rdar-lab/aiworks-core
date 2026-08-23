import logging

from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from ..models import (
    MemoryEntry,
)
from .views_utils import require_pro

logger = logging.getLogger(__name__)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def memory_list(request):
    """Return all memory entries for the current user."""
    entries = MemoryEntry.objects.filter(user=request.user).values(
        "id", "content", "created_at"
    )
    return Response(list(entries))


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def memory_import(request):
    """Import memories from another AI provider.

    Parses the user's provided export text and replaces all existing
    memory entries with the normalised, deduplicated set.
    """
    from ..logic.memory import import_memories

    imported_text = (request.data.get("text") or "").strip()
    if not imported_text:
        return Response(
            {"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    require_pro(request.user, "Import Memories")

    try:
        new_memories_count = import_memories(request.user, imported_text)

        return Response({"count": new_memories_count})
    except Exception as exp:
        logger.exception(f"Failed to process input memories, due to {exp}")
        return Response(
            {"error": "Failed to process memories"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def memory_delete(request, memory_id):
    """Delete a specific memory entry belonging to the current user."""
    try:
        entry = MemoryEntry.objects.get(pk=memory_id, user=request.user)
    except MemoryEntry.DoesNotExist:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    entry.delete()
    return Response({"status": "OK"}, status=status.HTTP_200_OK)
