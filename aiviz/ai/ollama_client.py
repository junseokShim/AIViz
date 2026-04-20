"""
Ollama HTTP client.

Wraps the Ollama REST API with:
- text generation (blocking)
- multimodal image+text generation (LLaVA-style)
- streaming generation
- health check and model listing
- graceful error handling

To swap the LLM backend: replace this file with a compatible
implementation that exposes the same public interface.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Generator, Optional

import httpx

from config import OLLAMA

logger = logging.getLogger("aiviz.ollama_client")


@dataclass
class OllamaResponse:
    text: str
    model: str
    done: bool
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class OllamaClient:
    """
    Minimal Ollama API client supporting text and multimodal generation.

    All public methods return OllamaResponse – they never raise.
    Callers check .ok before using .text.
    """

    def __init__(
        self,
        base_url: str = OLLAMA.base_url,
        default_model: str = OLLAMA.default_model,
        vision_model: str = OLLAMA.vision_model,
        timeout: int = OLLAMA.timeout,
        vision_timeout: int = OLLAMA.vision_timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.vision_model = vision_model
        self.timeout = timeout
        self.vision_timeout = vision_timeout

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: str = "",
        temperature: float = OLLAMA.temperature,
        max_tokens: int = OLLAMA.max_tokens,
    ) -> OllamaResponse:
        """
        Send a text prompt to Ollama and return the complete response.
        """
        model = model or self.default_model
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        return self._post(payload, model=model, timeout=self.timeout)

    # ------------------------------------------------------------------
    # Multimodal (vision) generation
    # ------------------------------------------------------------------

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = OLLAMA.temperature,
    ) -> OllamaResponse:
        """
        Send image + text prompt to a multimodal Ollama model (e.g. LLaVA).

        Requirements:
          - A vision-capable model must be pulled:
              ollama pull llava
          - The model name must be configured in OLLAMA_VISION_MODEL env var
            or passed explicitly via the `model` parameter.

        Args:
            prompt:      Text description of what to analyse.
            image_bytes: Raw bytes of the image (PNG, JPEG, etc.).
            model:       Override vision model name.
            temperature: Sampling temperature.

        Returns:
            OllamaResponse with .text populated on success.
        """
        model = model or self.vision_model
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64_image],   # Ollama multimodal API field
            "stream": False,
            "options": {"temperature": temperature},
        }
        return self._post(payload, model=model, timeout=self.vision_timeout)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: str = "",
        temperature: float = OLLAMA.temperature,
    ) -> Generator[str, None, None]:
        """
        Stream token chunks from Ollama.

        Yields one string chunk at a time.
        On error, yields an error message string and stops.
        """
        model = model or self.default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError:
            yield f"\n[Error: Cannot connect to Ollama at {self.base_url}]"
        except Exception as exc:
            yield f"\n[Stream error: {exc}]"

    # ------------------------------------------------------------------
    # Health & discovery
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally available models."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning(f"Could not list models: {exc}")
            return []

    def has_vision_model(self) -> bool:
        """Return True if the configured vision model is available locally."""
        models = self.list_models()
        return any(self.vision_model.split(":")[0] in m for m in models)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, payload: dict, model: str, timeout: int) -> OllamaResponse:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return OllamaResponse(
                text=data.get("response", "").strip(),
                model=model,
                done=data.get("done", True),
            )
        except httpx.ConnectError:
            return self._connection_error(model)
        except httpx.TimeoutException:
            return OllamaResponse(
                text="", model=model, done=False,
                error=f"Request timed out after {timeout}s. "
                      "Try increasing OLLAMA_TIMEOUT or using a smaller model.",
            )
        except httpx.HTTPStatusError as exc:
            return OllamaResponse(
                text="", model=model, done=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except Exception as exc:
            logger.exception("Unexpected Ollama error")
            return OllamaResponse(text="", model=model, done=False,
                                  error=f"Unexpected error: {exc}")

    def _connection_error(self, model: str) -> OllamaResponse:
        return OllamaResponse(
            text="", model=model, done=False,
            error=(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            ),
        )


# Module-level singleton
_default_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
