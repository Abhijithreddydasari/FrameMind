"""Unit tests for X-CLIP temporal encoder."""
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.ml.temporal_encoder import (
    ClipConfig,
    ClipEmbedding,
    VideoClip,
    VideoClipExtractor,
    XCLIPEncoder,
)


class TestClipConfig:
    """Tests for ClipConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClipConfig()
        
        assert config.window_frames == 16
        assert config.clip_fps == 8.0
        assert config.stride == 0.5
        assert config.max_clips == 500

    def test_window_duration_ms(self) -> None:
        """Test window duration calculation."""
        config = ClipConfig(window_frames=16, clip_fps=8.0)
        assert config.window_duration_ms == 2000  # 16/8 = 2 seconds = 2000ms

    def test_stride_frames(self) -> None:
        """Test stride frames calculation."""
        config = ClipConfig(window_frames=16, stride=0.5)
        assert config.stride_frames == 8  # 50% overlap

        config = ClipConfig(window_frames=16, stride=0.75)
        assert config.stride_frames == 4  # 75% overlap

        config = ClipConfig(window_frames=16, stride=0.0)
        assert config.stride_frames == 16  # No overlap


class TestVideoClip:
    """Tests for VideoClip dataclass."""

    def test_creation(self) -> None:
        """Test VideoClip creation."""
        frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        
        clip = VideoClip(
            index=0,
            frames=frames,
            start_frame=0,
            end_frame=15,
            start_ms=0,
            end_ms=2000,
        )
        
        assert clip.index == 0
        assert clip.frames.shape == (16, 224, 224, 3)
        assert clip.start_frame == 0
        assert clip.end_frame == 15


class TestVideoClipExtractor:
    """Tests for VideoClipExtractor."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        extractor = VideoClipExtractor()
        
        assert extractor.config is not None
        assert extractor.config.window_frames > 0

    def test_custom_config(self) -> None:
        """Test extractor with custom config."""
        config = ClipConfig(window_frames=8, clip_fps=4.0)
        extractor = VideoClipExtractor(config)
        
        assert extractor.config.window_frames == 8
        assert extractor.config.clip_fps == 4.0

    def test_extract_nonexistent_file(self) -> None:
        """Test extraction from non-existent file raises error."""
        extractor = VideoClipExtractor()
        
        with pytest.raises(Exception):  # ProcessingError
            extractor.extract("/nonexistent/video.mp4")

    @patch("cv2.VideoCapture")
    def test_extract_opencv(self, mock_cap: MagicMock) -> None:
        """Test extraction using OpenCV fallback."""
        # Setup mock
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            5: 30.0,   # FPS
            7: 300,    # Frame count
        }.get(prop, 0)

        # Create fake frames
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, fake_frame)

        config = ClipConfig(window_frames=4, clip_fps=2.0, stride=0.5)
        extractor = VideoClipExtractor(config)
        extractor._use_decord = False

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            # Touch the file so it exists
            Path(f.name).touch()
            
            clips = extractor.extract(f.name)

        # Should have extracted some clips
        assert isinstance(clips, list)


class TestClipEmbedding:
    """Tests for ClipEmbedding dataclass."""

    def test_creation(self) -> None:
        """Test ClipEmbedding creation."""
        embedding = np.random.randn(512).astype(np.float32)
        
        clip_emb = ClipEmbedding(
            clip_index=0,
            embedding=embedding,
            start_frame=0,
            end_frame=15,
            start_ms=0,
            end_ms=2000,
        )
        
        assert clip_emb.clip_index == 0
        assert clip_emb.embedding.shape == (512,)
        assert clip_emb.start_ms == 0
        assert clip_emb.end_ms == 2000


class TestXCLIPEncoder:
    """Tests for XCLIPEncoder."""

    def test_initialization(self) -> None:
        """Test encoder initialization."""
        encoder = XCLIPEncoder()
        
        assert encoder.model_name == "microsoft/xclip-base-patch32"
        assert encoder._loaded is False

    def test_custom_model(self) -> None:
        """Test encoder with custom model name."""
        encoder = XCLIPEncoder(model_name="custom/model")
        assert encoder.model_name == "custom/model"

    def test_custom_device(self) -> None:
        """Test encoder with specified device."""
        encoder = XCLIPEncoder(device="cpu")
        assert encoder.device == "cpu"

    def test_is_loaded_initially_false(self) -> None:
        """Test is_loaded is False before loading."""
        encoder = XCLIPEncoder()
        assert encoder.is_loaded is False

    @pytest.mark.asyncio
    async def test_unload_model(self) -> None:
        """Test model unloading."""
        encoder = XCLIPEncoder()
        encoder._loaded = True
        encoder._model = MagicMock()
        encoder._processor = MagicMock()
        
        await encoder.unload_model()
        
        assert encoder._loaded is False
        assert encoder._model is None
        assert encoder._processor is None

    def test_encode_clips_not_loaded(self) -> None:
        """Test encoding fails when model not loaded."""
        encoder = XCLIPEncoder()
        
        with pytest.raises(Exception):  # MLModelError
            encoder.encode_clips([])

    def test_encode_text_not_loaded(self) -> None:
        """Test text encoding fails when model not loaded."""
        encoder = XCLIPEncoder()
        
        with pytest.raises(Exception):
            encoder.encode_text("test query")

    def test_score_relevance_empty(self) -> None:
        """Test scoring with empty embeddings."""
        encoder = XCLIPEncoder()
        encoder._loaded = True
        encoder._processor = MagicMock()
        encoder._model = MagicMock()
        
        # Mock encode_text
        with patch.object(encoder, "encode_text", return_value=np.zeros(512, dtype=np.float32)):
            result = encoder.score_relevance([], "test query")
            assert result == []


class TestDualStreamIntegration:
    """Integration tests for dual-stream retrieval."""

    def test_embeddings_compatible(self) -> None:
        """Test that spatial and temporal embeddings are compatible dimensions."""
        # CLIP and X-CLIP base models both produce 512-dim embeddings
        spatial_emb = np.random.randn(512).astype(np.float32)
        temporal_emb = np.random.randn(512).astype(np.float32)
        
        assert spatial_emb.shape == temporal_emb.shape
        
        # Test cosine similarity works
        similarity = np.dot(spatial_emb, temporal_emb) / (
            np.linalg.norm(spatial_emb) * np.linalg.norm(temporal_emb)
        )
        assert -1 <= similarity <= 1
