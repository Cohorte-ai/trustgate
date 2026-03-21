"""TrustGate Quickstart -- Certify GPT-4.1-mini on 50 MCQ questions in 5 minutes."""

from theaios import trustgate
from theaios.trustgate.datasets import load_mmlu

# Load 50 MMLU abstract algebra questions (auto-downloaded, cached locally)
questions = load_mmlu(subjects=["abstract_algebra"], n=50)

# Ground truth labels come from the dataset itself
labels = {q.id: q.acceptable_answers[0] for q in questions if q.acceptable_answers}

# Certify
result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        sampling=trustgate.SamplingConfig(k_fixed=10),
        canonicalization=trustgate.CanonConfig(type="mcq"),
    ),
    questions=questions,
    labels=labels,
)

print(f"Reliability Level: {result.reliability_level:.1%}")
print(f"M* = {result.m_star}")
print(f"Coverage: {result.coverage:.3f}")
print(f"Capability Gap: {result.capability_gap:.1%}")
