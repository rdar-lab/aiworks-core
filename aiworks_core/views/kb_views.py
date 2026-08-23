import logging
from typing import cast

from django.db import IntegrityError, transaction
from django.db.models import Count
from rest_framework import viewsets, status
from rest_framework.decorators import (
    action,
)
from rest_framework.exceptions import (
    PermissionDenied,
    ValidationError as DRFValidationError,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .views_utils import require_pro, upload_files
from ..models import (
    AttachedFile,
    KnowledgeBase,
    User,
)
from ..serializers import (
    KnowledgeBaseSerializer,
)

logger = logging.getLogger(__name__)


class KnowledgeBaseViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user knowledge bases"""

    serializer_class = KnowledgeBaseSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (
            KnowledgeBase.objects.filter(user=self.request.user)
            .annotate(files_count=Count("kb_files"))
            .prefetch_related("kb_files")
        )

    def perform_create(self, serializer):
        require_pro(cast(User, self.request.user), "knowledge_base")
        try:
            with transaction.atomic():
                serializer.save(user=self.request.user)
        except IntegrityError:
            name = serializer.validated_data.get("name")
            if (
                    name
                    and KnowledgeBase.objects.filter(
                user=self.request.user, name=name
            ).exists()
            ):
                raise DRFValidationError(
                    {"error": "A knowledge base with this name already exists."}
                )
            raise

    def perform_update(self, serializer: KnowledgeBaseSerializer):
        if serializer.instance.user != self.request.user:
            raise PermissionDenied("Cannot update another user's knowledge base")
        try:
            with transaction.atomic():
                serializer.save()
        except IntegrityError:
            name = serializer.validated_data.get("name")
            if (
                    name
                    and KnowledgeBase.objects.filter(user=self.request.user, name=name)
                    .exclude(pk=serializer.instance.pk)
                    .exists()
            ):
                raise DRFValidationError(
                    {"error": "A knowledge base with this name already exists."}
                )
            raise

    def perform_destroy(self, instance: KnowledgeBase):
        if instance.user != self.request.user:
            raise PermissionDenied("Cannot delete another user's knowledge base")
        instance.delete()

    def destroy(self, request, *args, **kwargs):
        super().destroy(request, *args, **kwargs)
        return Response({"status": "OK"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def upload_files(self, request, *_args, **_kwargs):
        """Upload files to a knowledge base."""
        require_pro(request.user, "knowledge_base")
        kb = self.get_object()
        created, errors = upload_files(
            request, additional_file_fields={"knowledge_base": kb}
        )
        response_data = {}
        if created:
            response_data["files"] = created
        if errors:
            response_data["parse_errors"] = errors
        return Response(
            response_data,
            status=status.HTTP_207_MULTI_STATUS if errors else status.HTTP_201_CREATED,
        )

    # noinspection PyUnusedLocal
    @action(detail=True, methods=["delete"], url_path="files/(?P<file_id>[^/.]+)")
    def remove_file(self, request, file_id=None, *_args, **_kwargs):
        """Remove a file from a knowledge base."""
        kb = self.get_object()
        try:
            kb_file = AttachedFile.objects.get(id=file_id, knowledge_base=kb)
        except AttachedFile.DoesNotExist:
            return Response(
                {"error": "File not found"}, status=status.HTTP_404_NOT_FOUND
            )
        kb_file.delete()
        logger.info("kb.remove_file | kb=%s | file=%s", kb.id, file_id)
        return Response({"status": "OK"}, status=status.HTTP_200_OK)
