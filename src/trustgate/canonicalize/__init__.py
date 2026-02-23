"""Canonicalization base class, registry, and shared preprocessing."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

# Patterns stripped during preprocessing
_CODE_FENCE_RE = re.compile(r"```[\w]*\n?(.*?)```", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^(?:Sure[,!.]?\s*|Certainly[,!.]?\s*|Of course[,!.]?\s*|"
    r"I think\s+|I believe\s+|Let me think[.\s]*|"
    r"Here(?:'s| is) (?:the |my )?(?:answer|response|solution)[:\s]*)",
    re.IGNORECASE,
)


class Canonicalizer(ABC):
    """Base class for all canonicalizers."""

    @abstractmethod
    def canonicalize(self, question: str, answer: str) -> str:
        """Map a raw LLM answer to a canonical form.

        Returns a canonical string.  Two answers that mean the same thing
        should produce the same canonical string.
        """

    def validate(self, canonical: str) -> bool:
        """Optional: check that the canonical form is well-formed."""
        return True

    def preprocess(self, answer: str) -> str:
        """Shared preprocessing applied before canonicalization.

        1. Strip leading/trailing whitespace
        2. Remove markdown code fences (keep inner content)
        3. Remove common LLM preambles
        4. Normalize unicode (NFC)
        """
        if not answer:
            return ""

        text = answer.strip()

        # Remove code fences but keep the content inside them
        text = _CODE_FENCE_RE.sub(r"\1", text)

        # Strip common preambles
        text = _PREAMBLE_RE.sub("", text).strip()

        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        return text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Canonicalizer]] = {}


def register_canonicalizer(name: str) -> Any:  # noqa: ANN401
    """Decorator to register a canonicalizer class by name.

    Usage::

        @register_canonicalizer("numeric")
        class NumericCanonicalizer(Canonicalizer):
            ...
    """

    def decorator(cls: type[Canonicalizer]) -> type[Canonicalizer]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_canonicalizer(name: str, **kwargs: Any) -> Canonicalizer:  # noqa: ANN401
    """Instantiate a registered canonicalizer by name."""
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown canonicalizer '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_canonicalizers() -> list[str]:
    """Return all registered canonicalizer names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Eagerly import built-in canonicalizer modules so decorators run.
# ---------------------------------------------------------------------------
import trustgate.canonicalize.code_exec as _code_exec  # noqa: E402, F401
import trustgate.canonicalize.embedding as _embedding  # noqa: E402, F401
import trustgate.canonicalize.llm_judge as _llm_judge  # noqa: E402, F401
import trustgate.canonicalize.mcq as _mcq  # noqa: E402, F401
import trustgate.canonicalize.numeric as _numeric  # noqa: E402, F401
