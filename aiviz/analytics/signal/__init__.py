"""
Advanced signal analysis package.

Provides modular, UI-independent implementations of:
- FFT (wrapper around existing frequency.py)
- PSD / Welch
- STFT
- CWT (Continuous Wavelet Transform)
- S-Transform
- Band Power
- Envelope Spectrum
- Cepstrum
"""
from .fft import run_fft
from .psd import compute_psd, PSDResult
from .stft import compute_stft, STFTResult
from .cwt import compute_cwt, CWTResult, HAS_PYWT
from .s_transform import compute_s_transform, STransformResult
from .band_power import compute_band_power, BandPowerResult
from .envelope import compute_envelope_spectrum, compute_cepstrum, EnvelopeResult, CepstrumResult

__all__ = [
    "run_fft",
    "compute_psd", "PSDResult",
    "compute_stft", "STFTResult",
    "compute_cwt", "CWTResult", "HAS_PYWT",
    "compute_s_transform", "STransformResult",
    "compute_band_power", "BandPowerResult",
    "compute_envelope_spectrum", "compute_cepstrum", "EnvelopeResult", "CepstrumResult",
]
