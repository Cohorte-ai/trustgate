"""TrustGate — Black-box AI reliability certification."""

__version__ = "0.1.0"

from trustgate.calibration import calibrate, compute_profile
from trustgate.canonicalize import Canonicalizer, get_canonicalizer, register_canonicalizer
from trustgate.certification import certify, certify_async
from trustgate.config import load_config
from trustgate.sampler import sample
from trustgate.types import (
    CalibrationConfig,
    CanonConfig,
    CertificationResult,
    EndpointConfig,
    Question,
    SamplingConfig,
    TrustGateConfig,
)

__all__ = [
    "certify",
    "certify_async",
    "calibrate",
    "compute_profile",
    "sample",
    "load_config",
    "Canonicalizer",
    "register_canonicalizer",
    "get_canonicalizer",
    "TrustGateConfig",
    "CertificationResult",
    "Question",
    "EndpointConfig",
    "SamplingConfig",
    "CanonConfig",
    "CalibrationConfig",
]
