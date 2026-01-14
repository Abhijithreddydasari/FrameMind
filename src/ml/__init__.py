"""ML module - Core CV/ML components for intelligent frame selection."""
from src.ml.clip_scorer import CLIPScorer
from src.ml.frame_selector import FrameSelector
from src.ml.shot_detector import ShotDetector

__all__ = ["CLIPScorer", "ShotDetector", "FrameSelector"]
