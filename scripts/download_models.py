#!/usr/bin/env python3
"""Download and cache ML models for offline use.

This script pre-downloads the CLIP model so it's available
without network access during inference.

Usage:
    python scripts/download_models.py
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_clip_model() -> None:
    """Download CLIP model and processor."""
    from transformers import CLIPModel, CLIPProcessor
    
    from src.core.config import settings
    
    print(f"Downloading CLIP model: {settings.clip_model}")
    
    # This will download and cache the model
    processor = CLIPProcessor.from_pretrained(settings.clip_model)
    model = CLIPModel.from_pretrained(settings.clip_model)
    
    print(f"Model downloaded successfully!")
    print(f"Cache location: ~/.cache/huggingface/")
    
    # Verify model works
    print("Verifying model...")
    inputs = processor(
        text=["test"],
        images=None,
        return_tensors="pt",
        padding=True,
    )
    text_features = model.get_text_features(**inputs)
    print(f"Text embedding shape: {text_features.shape}")
    print("Verification complete!")


def main() -> None:
    """Main entry point."""
    print("=" * 50)
    print("FrameMind Model Downloader")
    print("=" * 50)
    print()
    
    try:
        download_clip_model()
        print()
        print("All models downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
