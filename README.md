<div align="center">
  <a href="https://cohorte-ai.github.io/trustgate/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/TheAIOS-TrustGate-darkmode.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/TheAIOS-TrustGate.svg">
      <img alt="TrustGate" src=".github/images/TheAIOS-TrustGate.svg" width="60%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>Know if your AI is ready to ship — one number, one guarantee.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://pypi.org/project/theaios-trustgate/" target="_blank"><img src="https://img.shields.io/pypi/v/theaios-trustgate?label=%20" alt="Version"></a>
  <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="https://cohorte-ai.github.io/trustgate/" target="_blank"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
</div>

<br>

TrustGate certifies the reliability of any AI endpoint — LLMs, agents, RAG pipelines, or any system you can ask a question to. It uses self-consistency sampling and conformal prediction to produce a single **reliability level** (e.g., 94.6%) backed by a formal statistical guarantee. Not a vibe, not a leaderboard score — a mathematical proof.

**What's included:**

- **Self-consistency sampling** — ask the same question K times, measure agreement
- **Conformal calibration** — formal coverage guarantee, distribution-free
- **Human calibration** — shareable HTML questionnaire for domain experts (no server needed)
- **Runtime trust layer** — wrap any endpoint with reliability metadata
- **Sequential stopping** — Hoeffding bounds reduce API costs by ~50%
- **Profile diagnostics** — automatic detection of canonicalization failures

> [!NOTE]
> Part of the [theaios](https://github.com/Cohorte-ai) ecosystem. Install with `pip install theaios-trustgate`.

## Quickstart

```bash
pip install theaios-trustgate
```

```python
from theaios import trustgate

result = trustgate.certify(config_path="trustgate.yaml")
print(result.reliability_level)  # 0.946
```

The pipeline: sample K responses → canonicalize → calibrate with conformal prediction → get a reliability level with a guarantee. Works with any provider (OpenAI, Anthropic, self-hosted), any task type, fully black-box.

> [!TIP]
> For the full theory, see our paper: *Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration* (Mouzouni, 2026).

## Three Ways to Use TrustGate

### 1. Deployment gate — certify before shipping

```bash
trustgate certify --yes
# Exit code 0 = PASS, 1 = FAIL
```

```
     TrustGate Certification Result
┌──────────────────────┬──────────┐
│ Reliability Level    │ 94.6%    │
│ M* (prediction set)  │ 1        │
│ Empirical Coverage   │ 0.956    │
│ Capability Gap       │ 2.4%     │
│ Status               │ PASS     │
└──────────────────────┴──────────┘
```

### 2. Runtime trust layer — confidence on every query

```python
from theaios.trustgate import TrustGate, certify

result = certify(config_path="trustgate.yaml")
gate = TrustGate(config=config, certification=result)

# Passthrough (1 API call): attaches reliability metadata
response = gate.query("What is the treatment for X?")
response.reliability_level  # 0.946

# Sampled (K API calls): per-query prediction set
gate = TrustGate(config=config, certification=result, mode="sampled")
response = gate.query("What is the treatment for X?")
response.prediction_set  # ["Aspirin + PCI"]
response.consensus       # 0.8
```

### 3. Human calibration — no ground truth needed

Generate a questionnaire, share it with a domain expert, certify with their labels:

```bash
trustgate calibrate --export questionnaire.html
# Share via email/Slack → reviewer opens in browser → downloads labels.json
trustgate certify --ground-truth labels.json
```

## Works With Any Endpoint

LLMs, agents, RAG pipelines — anything with an HTTP API:

```yaml
# LLM
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1"
  api_key_env: "OPENAI_API_KEY"

# Generic agent / RAG / custom API
endpoint:
  url: "https://my-agent.example.com/api/ask"
  temperature: null
  request_template:
    query: "{{question}}"
  response_path: "answer"
  cost_per_request: 0.03
```

## Certify at the Decision Point

Self-consistency works on **short, structured outputs**. For systems that produce long outputs, certify at the decision point:

| System | Long output | Certify on |
|--------|------------|------------|
| SQL agent | English report | The SQL query |
| Medical triage | Patient summary | Triage category (1–5) |
| Legal review | Contract analysis | Conclusion (approve/reject) |
| Code agent | Full implementation | Test pass/fail |

TrustGate warns you automatically when canonicalization is failing.

## Pre-flight Cost Estimate

Before spending money, TrustGate shows the cost/reliability tradeoff:

```
     Cost / Reliability Tradeoff
┌────┬───────────┬──────────┬────────────┐
│  K │ Est. Cost │ Max Cost │ Resolution │
│  3 │ $9.00     │ $18.00   │   coarse   │
│  5 │ $15.00    │ $30.00   │  moderate  │
│ 10←│ $30.00    │ $60.00   │    fine    │
│ 20 │ $60.00    │ $120.00  │    fine    │
└────┴───────────┴──────────┴────────────┘
Proceed? [Y/n]:
```

---

## Documentation

- [cohorte-ai.github.io/trustgate](https://cohorte-ai.github.io/trustgate/) — Full documentation
- [Concepts](https://cohorte-ai.github.io/trustgate/concepts/) — How self-consistency + conformal prediction works
- [Configuration](https://cohorte-ai.github.io/trustgate/configuration/) — Full YAML reference
- [CLI Reference](https://cohorte-ai.github.io/trustgate/cli/) — All commands and flags
- [Human Calibration](https://cohorte-ai.github.io/trustgate/human-calibration/) — Questionnaire and review UI
- [API Reference](https://cohorte-ai.github.io/trustgate/api-reference/) — Python API

## Additional Resources

- **[Examples](examples/)** — Working certification scripts
- **[FAQ](https://cohorte-ai.github.io/trustgate/faq/)** — Common questions
- **[Paper](https://arxiv.org/)** — The research behind TrustGate

---

## Why TrustGate?

- **Formal guarantee** — conformal coverage bound, not a heuristic score
- **Black-box** — no model internals, no logprobs, just API access
- **Any endpoint** — LLMs, agents, RAG, custom APIs
- **Human-in-the-loop** — shareable questionnaire, no server needed
- **Cost-aware** — pre-flight estimates, sequential stopping saves ~50%
- **Production-ready** — passthrough mode, CI/CD gating, periodic recalibration

---

## Citation

```bibtex
@article{mouzouni2026trustgate,
  title={TrustGate: Black-Box AI Reliability Certification via
         Self-Consistency Sampling and Conformal Calibration},
  author={Mouzouni, Charafeddine},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
