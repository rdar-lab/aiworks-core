"""
Tests for the video_generator module.
"""

from django.test import TestCase

from ..logic.video_generator import VideoGeneratorModel


class VideoGeneratorModelTests(TestCase):
    def test_video_generator_model_protocol_exists(self):
        """VideoGeneratorModel protocol is importable and defines generate_video."""
        self.assertTrue(hasattr(VideoGeneratorModel, "generate_video"))

    def test_generate_video_method_signature(self):
        """VideoGeneratorModel.generate_video has the expected signature."""
        import inspect
        sig = inspect.signature(VideoGeneratorModel.generate_video)
        params = list(sig.parameters.keys())
        self.assertIn("messages", params)
        self.assertIn("kwargs", params)

    def test_generate_video_is_abc_with_abstract_method(self):
        """VideoGeneratorModel is an ABC with generate_video as abstract method."""
        from abc import ABC
        self.assertTrue(issubclass(VideoGeneratorModel, ABC))
        self.assertTrue(hasattr(VideoGeneratorModel, "generate_video"))
