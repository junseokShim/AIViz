"""
AIViz AI Agent layer.

Orchestrates: data context → prompt template → Ollama client → AgentResult.
Stateless – context is passed per call.
All methods provide meaningful fallback responses when Ollama is unavailable.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from PIL import Image

from aiviz.ai.ollama_client import OllamaClient, OllamaResponse, get_client
from aiviz.ai import prompts
from aiviz.utils.helpers import df_to_context_string, truncate_str
from config import OLLAMA

logger = logging.getLogger("aiviz.agent")


@dataclass
class AgentResult:
    answer: str
    model: str
    error: Optional[str] = None
    is_fallback: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class AnalysisAgent:
    """
    Main AI analysis agent.

    Wraps all LLM calls with:
    - offline-health check
    - fallback computation when AI unavailable
    - error surfacing without exceptions
    """

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or get_client()
        self.model = OLLAMA.default_model

    # ------------------------------------------------------------------
    # Text analysis methods
    # ------------------------------------------------------------------

    def summarize_dataset(self, df: pd.DataFrame, question: str = "") -> AgentResult:
        context = truncate_str(df_to_context_string(df), max_len=3000)
        prompt = prompts.data_summary_prompt(context, question)
        return self._call(prompt, fallback=self._fallback_summary(df))

    def explain_timeseries(
        self, col_name: str, stats: dict, anomaly_count: int
    ) -> AgentResult:
        slope = stats.get("trend_slope", 0)
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        prompt = prompts.timeseries_analysis_prompt(col_name, stats, anomaly_count, direction)
        return self._call(
            prompt, fallback=self._fallback_timeseries(col_name, stats, anomaly_count)
        )

    def explain_fft(self, fft_stats: dict, col_name: str) -> AgentResult:
        prompt = prompts.frequency_analysis_prompt(fft_stats, col_name)
        return self._call(prompt, fallback=self._fallback_fft(fft_stats, col_name))

    def explain_image_stats(self, image_info: dict, question: str = "") -> AgentResult:
        ch_stats = image_info.get("channel_stats")
        if ch_stats is not None and hasattr(ch_stats, "to_string"):
            image_info = dict(image_info)
            image_info["channel_stats_text"] = ch_stats.to_string(index=False)
        prompt = prompts.image_analysis_prompt(image_info, question)
        return self._call(prompt, fallback=self._fallback_image(image_info))

    def explain_forecast(
        self, col_name: str, method: str, metrics: dict, horizon: int
    ) -> AgentResult:
        prompt = prompts.forecast_prompt(col_name, method, metrics, horizon)
        fallback = (
            f"Forecast ({method}) for '{col_name}': "
            f"RMSE={metrics.get('rmse', 'N/A')}, MAE={metrics.get('mae', 'N/A')}. "
            "(AI assistant unavailable)"
        )
        return self._call(prompt, fallback=fallback)

    def suggest_charts(self, df: pd.DataFrame) -> AgentResult:
        context = truncate_str(df_to_context_string(df, max_rows=5), max_len=2000)
        prompt = prompts.chart_suggestion_prompt(context)
        return self._call(
            prompt,
            fallback="Consider: line chart for trends, histogram for distributions, scatter for correlations."
        )

    def ask(self, df: pd.DataFrame, question: str) -> AgentResult:
        context = truncate_str(df_to_context_string(df), max_len=3000)
        prompt = prompts.general_question_prompt(context, question)
        return self._call(prompt, fallback=f"AI unavailable. Your question was: {question}")

    # ------------------------------------------------------------------
    # Multimodal (vision) analysis
    # ------------------------------------------------------------------

    def describe_image_visual(
        self,
        img: Image.Image,
        question: str = "",
    ) -> AgentResult:
        """
        Send the actual image pixels to a LLaVA-style Ollama vision model.

        Requires a vision model (e.g. `llava`) to be pulled locally.
        Falls back gracefully if vision model is not available.
        """
        if not self.client.is_healthy():
            return AgentResult(
                answer="Ollama unavailable. Start with `ollama serve`.",
                model="(offline)", is_fallback=True,
                error="Ollama not reachable"
            )

        if not self.client.has_vision_model():
            return AgentResult(
                answer=(
                    f"Vision model `{self.client.vision_model}` is not available.\n"
                    f"Pull it with: `ollama pull {self.client.vision_model}`\n\n"
                    "You can still use the text-based image stats analysis below."
                ),
                model=self.client.vision_model,
                is_fallback=True,
                error="Vision model not found",
            )

        # Convert image to JPEG bytes for transmission
        try:
            buf = io.BytesIO()
            img_rgb = img.convert("RGB")
            img_rgb.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()
        except Exception as exc:
            return AgentResult(
                answer=f"Image encoding error: {exc}",
                model=self.client.vision_model,
                is_fallback=True, error=str(exc)
            )

        prompt = prompts.multimodal_image_prompt(question)
        resp: OllamaResponse = self.client.generate_with_image(
            prompt, image_bytes, model=self.client.vision_model
        )

        if not resp.ok:
            return AgentResult(
                answer=f"Vision model error: {resp.error}",
                model=resp.model, is_fallback=True, error=resp.error
            )

        return AgentResult(answer=resp.text, model=resp.model)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str, fallback: str = "") -> AgentResult:
        if not self.client.is_healthy():
            logger.warning("Ollama not available – returning fallback")
            return AgentResult(
                answer=fallback or "AI assistant unavailable. Run `ollama serve`.",
                model="(offline)", error="Ollama not reachable", is_fallback=True,
            )

        resp: OllamaResponse = self.client.generate(prompt, model=self.model)
        if not resp.ok:
            return AgentResult(
                answer=fallback or f"AI error: {resp.error}",
                model=resp.model, error=resp.error, is_fallback=True,
            )
        return AgentResult(answer=resp.text, model=resp.model)

    # ------------------------------------------------------------------
    # Fallback generators (no LLM needed)
    # ------------------------------------------------------------------

    def _fallback_summary(self, df: pd.DataFrame) -> str:
        from aiviz.analytics.summary import compute_summary
        s = compute_summary(df)
        return (
            f"Dataset: {s.schema.n_rows} rows × {s.schema.n_cols} columns. "
            f"Numeric: {', '.join(s.schema.numeric_cols()) or 'none'}. "
            f"Missing: {sum(c.null_count for c in s.schema.columns)} values. "
            "(AI unavailable – showing computed summary)"
        )

    def _fallback_timeseries(self, col: str, stats: dict, anomalies: int) -> str:
        slope = stats.get("trend_slope", 0)
        dir_ = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        return (
            f"Column '{col}': trend {dir_} (slope={slope:.4g}), "
            f"{anomalies} anomalies. Mean={stats.get('mean', 0):.4g}. "
            "(AI unavailable)"
        )

    def _fallback_fft(self, stats: dict, col: str) -> str:
        return (
            f"FFT of '{col}': dominant freq={stats.get('dominant_freq', 0):.4g} Hz, "
            f"power={stats.get('total_power', 0):.4g}. (AI unavailable)"
        )

    def _fallback_image(self, info: dict) -> str:
        return (
            f"Image {info.get('width')}×{info.get('height')} px, "
            f"mode={info.get('mode')}, channels={info.get('n_channels')}. "
            "(AI unavailable)"
        )
