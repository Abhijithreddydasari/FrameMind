"""Tests for shot detection module."""
import numpy as np
import pytest

from src.ml.shot_detector import HistogramConfig, ShotDetector


class TestShotDetector:
    """Test cases for ShotDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        detector = ShotDetector()
        assert detector.config.bins == 64
        assert detector.config.threshold > 0

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = HistogramConfig(bins=32, threshold=0.5)
        detector = ShotDetector(config=config)
        assert detector.config.bins == 32
        assert detector.config.threshold == 0.5

    def test_compute_histogram(self, sample_frame: np.ndarray) -> None:
        """Test histogram computation."""
        detector = ShotDetector()
        hist = detector.compute_histogram(sample_frame)
        
        assert hist is not None
        assert len(hist) == 64 * 3  # bins * channels
        assert np.isclose(hist.sum(), 1.0, atol=0.01)  # Normalized

    def test_compare_identical_histograms(self) -> None:
        """Test that identical histograms have zero distance."""
        detector = ShotDetector()
        hist = np.ones(192, dtype=np.float32) / 192
        
        distance = detector.compare_histograms(hist, hist)
        assert distance < 0.001

    def test_compare_different_histograms(self) -> None:
        """Test that different histograms have non-zero distance."""
        detector = ShotDetector()
        hist1 = np.zeros(192, dtype=np.float32)
        hist1[:96] = 1.0 / 96
        
        hist2 = np.zeros(192, dtype=np.float32)
        hist2[96:] = 1.0 / 96
        
        distance = detector.compare_histograms(hist1, hist2)
        assert distance > 0.5

    def test_detect_no_boundaries_similar_frames(self) -> None:
        """Test that similar frames don't trigger boundaries."""
        detector = ShotDetector(HistogramConfig(threshold=0.5))
        
        # Create similar frames
        frames = [
            np.full((100, 100, 3), fill_value=128, dtype=np.uint8)
            for _ in range(5)
        ]
        
        boundaries = detector.detect_from_frames(frames)
        assert len(boundaries) == 0

    def test_detect_boundaries_different_frames(self) -> None:
        """Test that different frames trigger boundaries."""
        # Disable adaptive threshold to ensure consistent behavior
        detector = ShotDetector(HistogramConfig(
            threshold=0.1, 
            min_scene_length=1,
            adaptive_threshold=False,
        ))
        
        # Create very different frames
        frames = [
            np.full((100, 100, 3), fill_value=0, dtype=np.uint8),
            np.full((100, 100, 3), fill_value=255, dtype=np.uint8),
        ]
        
        boundaries = detector.detect_from_frames(frames)
        assert len(boundaries) >= 1

    def test_streaming_detection(self, sample_frame: np.ndarray) -> None:
        """Test streaming mode detection."""
        detector = ShotDetector()
        
        # First frame should not return boundary
        result1 = detector.detect_streaming(sample_frame, 0)
        assert result1 is None
        
        # Similar frame should not trigger
        result2 = detector.detect_streaming(sample_frame, 1)
        assert result2 is None
        
        # Reset state
        detector.reset_streaming_state()

    def test_min_scene_length_enforced(self) -> None:
        """Test that minimum scene length is enforced."""
        config = HistogramConfig(threshold=0.01, min_scene_length=5)
        detector = ShotDetector(config)
        
        # Create alternating black/white frames
        frames = []
        for i in range(10):
            color = 255 if i % 2 else 0
            frames.append(np.full((100, 100, 3), fill_value=color, dtype=np.uint8))
        
        boundaries = detector.detect_from_frames(frames)
        
        # Should have fewer boundaries due to min_scene_length
        for i, b in enumerate(boundaries[1:], 1):
            prev = boundaries[i - 1]
            assert b.frame_index - prev.frame_index >= config.min_scene_length
