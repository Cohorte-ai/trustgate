"""TrustGate — Black-box AI reliability certification."""

__version__ = "0.1.0"

from theaios.trustgate.calibration import calibrate, compute_profile, diagnose_profiles
from theaios.trustgate.canonicalize import Canonicalizer, get_canonicalizer, register_canonicalizer
from theaios.trustgate.certification import (
    certify,
    certify_async,
    sample_and_profile,
    sample_and_rank,
)
from theaios.trustgate.config import load_config
from theaios.trustgate.sampler import sample
from theaios.trustgate.types import (
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
    "sample_and_profile",
    "sample_and_rank",
    "calibrate",
    "compute_profile",
    "diagnose_profiles",
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
