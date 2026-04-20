"""
AIViz global configuration.

All environment-specific and model-specific settings live here.
Swap models, endpoints, or timeouts without touching business logic.
"""

from dataclasses import dataclass, field
import os


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    # Vision model for multimodal image analysis (e.g. llava, llava:13b)
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava")
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    vision_timeout: int = int(os.getenv("OLLAMA_VISION_TIMEOUT", "120"))
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class AppConfig:
    name: str = "AIViz"
    version: str = "0.2.0"
    max_file_mb: int = 500
    default_sample_rows: int = 500
    supported_tabular: tuple = (".csv", ".xlsx", ".xls", ".json", ".parquet")
    supported_image: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    fft_default_window: str = "hann"
    timeseries_default_window: int = 10


@dataclass
class ForecastConfig:
    default_horizon: int = 30
    default_method: str = "Holt-Winters"   # "Holt-Winters" | "ARIMA" | "Simple"
    available_methods: tuple = ("Holt-Winters", "ARIMA", "Simple ES")


# Module-level singletons used across the app
OLLAMA = OllamaConfig()
APP = AppConfig()
FORECAST = ForecastConfig()
