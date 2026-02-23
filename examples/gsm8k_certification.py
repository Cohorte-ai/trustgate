"""Certify a model on GSM8K math problems with sequential stopping."""

import trustgate
from trustgate.datasets import load_gsm8k
from trustgate.reporting.json_export import export_json

# Load 200 GSM8K math problems
questions = load_gsm8k(n=200)
labels = {q.id: q.acceptable_answers[0] for q in questions if q.acceptable_answers}

# Certify with sequential stopping (saves ~50% of API costs)
result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7,
        ),
        sampling=trustgate.SamplingConfig(
            k_max=20,
            sequential_stopping=True,
            delta=0.05,
        ),
        canonicalization=trustgate.CanonConfig(type="numeric"),
    ),
    questions=questions,
    labels=labels,
)

# Export results to JSON
export_json(result, path="gsm8k_result.json")
print(f"Reliability: {result.reliability_level:.1%}")
print(f"Average K used: {result.k_used}")
print(f"Estimated cost: ${result.api_cost_estimate:.2f}")
print("Results saved to gsm8k_result.json")
