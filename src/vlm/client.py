"""Abstract VLM client with provider implementations."""
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.core.exceptions import VLMError, VLMRateLimitError, VLMTimeoutError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VLMResponse:
    """Response from VLM API."""

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str


class VLMClient(ABC):
    """Abstract VLM client interface.
    
    Provides a consistent interface for different VLM providers
    (OpenAI, Anthropic, etc.) with built-in retry logic.
    """

    @abstractmethod
    async def analyze_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze a single image.
        
        Args:
            image: Image data (bytes, base64 string, or path)
            prompt: Analysis prompt
            **kwargs: Provider-specific options
            
        Returns:
            VLM response with analysis
        """
        ...

    @abstractmethod
    async def analyze_images(
        self,
        images: list[bytes | str | Path],
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze multiple images together.
        
        Args:
            images: List of images
            prompt: Analysis prompt
            **kwargs: Provider-specific options
            
        Returns:
            VLM response with analysis
        """
        ...

    @classmethod
    def create(cls, provider: str | None = None) -> "VLMClient":
        """Factory method to create appropriate client.
        
        Args:
            provider: Provider name (openai, anthropic)
            
        Returns:
            Configured VLM client
        """
        provider = provider or settings.vlm_provider

        if provider == "openai":
            return OpenAIVLMClient()
        elif provider == "anthropic":
            return AnthropicVLMClient()
        else:
            raise VLMError(f"Unknown VLM provider: {provider}")


class OpenAIVLMClient(VLMClient):
    """OpenAI GPT-4V client implementation."""

    def __init__(self) -> None:
        self.api_key = settings.vlm_api_key
        self.model = settings.vlm_model
        self.base_url = "https://api.openai.com/v1"
        self.timeout = settings.vlm_timeout

    def _encode_image(self, image: bytes | str | Path) -> str:
        """Encode image to base64 data URL."""
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = f.read()

        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

        # Assume already base64 encoded
        return image

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def analyze_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze image with GPT-4V."""
        if not self.api_key:
            raise VLMError("OpenAI API key not configured")

        image_url = self._encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        return await self._call_api(messages, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def analyze_images(
        self,
        images: list[bytes | str | Path],
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze multiple images with GPT-4V."""
        if not self.api_key:
            raise VLMError("OpenAI API key not configured")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for image in images:
            image_url = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        messages = [{"role": "user", "content": content}]

        return await self._call_api(messages, **kwargs)

    async def _call_api(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> VLMResponse:
        """Make API call to OpenAI."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 1024),
                    },
                )

                if response.status_code == 429:
                    raise VLMRateLimitError("OpenAI rate limit exceeded")

                response.raise_for_status()
                data = response.json()

                return VLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=data["choices"][0].get("finish_reason", ""),
                )

            except httpx.TimeoutException:
                raise VLMTimeoutError("OpenAI request timed out")
            except httpx.HTTPStatusError as e:
                raise VLMError(f"OpenAI API error: {e.response.text}")


class AnthropicVLMClient(VLMClient):
    """Anthropic Claude client implementation."""

    def __init__(self) -> None:
        self.api_key = settings.vlm_api_key
        self.model = "claude-3-opus-20240229"  # Default to Opus for vision
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = settings.vlm_timeout

    def _encode_image(self, image: bytes | str | Path) -> dict[str, Any]:
        """Encode image for Anthropic API."""
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = f.read()

        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("utf-8")
            return {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            }

        # Assume already base64
        return {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def analyze_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze image with Claude."""
        if not self.api_key:
            raise VLMError("Anthropic API key not configured")

        image_data = self._encode_image(image)

        content = [
            {"type": "image", "source": image_data},
            {"type": "text", "text": prompt},
        ]

        return await self._call_api(content, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def analyze_images(
        self,
        images: list[bytes | str | Path],
        prompt: str,
        **kwargs: Any,
    ) -> VLMResponse:
        """Analyze multiple images with Claude."""
        if not self.api_key:
            raise VLMError("Anthropic API key not configured")

        content: list[dict[str, Any]] = []

        for image in images:
            image_data = self._encode_image(image)
            content.append({"type": "image", "source": image_data})

        content.append({"type": "text", "text": prompt})

        return await self._call_api(content, **kwargs)

    async def _call_api(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> VLMResponse:
        """Make API call to Anthropic."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": kwargs.get("max_tokens", 1024),
                        "messages": [{"role": "user", "content": content}],
                    },
                )

                if response.status_code == 429:
                    raise VLMRateLimitError("Anthropic rate limit exceeded")

                response.raise_for_status()
                data = response.json()

                return VLMResponse(
                    content=data["content"][0]["text"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=data.get("stop_reason", ""),
                )

            except httpx.TimeoutException:
                raise VLMTimeoutError("Anthropic request timed out")
            except httpx.HTTPStatusError as e:
                raise VLMError(f"Anthropic API error: {e.response.text}")
