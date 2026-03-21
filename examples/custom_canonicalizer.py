"""Example: Custom canonicalizer for medical entity extraction."""

import re

from theaios import trustgate
from theaios.trustgate import Canonicalizer, register_canonicalizer


@register_canonicalizer("medical")
class MedicalCanonicalizer(Canonicalizer):
    """Extract the first capitalized medical term from a response."""

    def canonicalize(self, question: str, answer: str) -> str:
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", answer)
        return terms[0].lower() if terms else "unknown"


# Use it in a certification run
result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
        ),
        sampling=trustgate.SamplingConfig(k_fixed=10),
        canonicalization=trustgate.CanonConfig(type="medical"),
    ),
    questions=[
        trustgate.Question(
            id="q1",
            text="What causes type 2 diabetes?",
            acceptable_answers=["insulin resistance"],
        ),
        trustgate.Question(
            id="q2",
            text="What is the treatment for hypertension?",
            acceptable_answers=["antihypertensive medication"],
        ),
    ],
    labels={"q1": "insulin resistance", "q2": "antihypertensive medication"},
)

print(f"Reliability Level: {result.reliability_level:.1%}")
