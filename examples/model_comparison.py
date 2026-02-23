"""Compare GPT-4.1 vs GPT-4.1-mini on MMLU."""

import trustgate
from trustgate.comparison import compare
from trustgate.datasets import load_mmlu
from trustgate.reporting.console import print_comparison_result

# Load 100 MMLU questions
questions = load_mmlu(n=100)
labels = {q.id: q.acceptable_answers[0] for q in questions if q.acceptable_answers}

# Compare two models side-by-side
results = compare(
    models=["gpt-4.1", "gpt-4.1-mini"],
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            api_key_env="OPENAI_API_KEY",
        ),
        sampling=trustgate.SamplingConfig(k_fixed=10),
        canonicalization=trustgate.CanonConfig(type="mcq"),
    ),
    questions=questions,
    labels=labels,
)

print_comparison_result(results)
