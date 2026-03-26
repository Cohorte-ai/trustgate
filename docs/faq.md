# Frequently Asked Questions

Common questions and troubleshooting for TrustGate.

---

## 1. What does "reliability level" mean?

The reliability level is the largest `1 - alpha` for which the conformal
coverage guarantee holds on the held-out test set. It answers the question:
"With what confidence can I trust this model's top answer?"

For example, a reliability level of **0.90** (90%) means that the model's
prediction set of size M\* contains the correct answer at least 90% of the time,
validated on unseen test data using conformal prediction theory.

Higher is better. A reliability level of 0.95 is a stronger guarantee than 0.90.

---

## 2. How many questions do I need?

**Minimum:** 100 questions (split into 50 calibration + 50 test).

**Recommended:** 500 or more. The conformal guarantee becomes tighter and more
meaningful with larger sample sizes. With 500 questions (250 calibration + 250
test), you get statistically robust coverage estimates.

The calibration/test split is configured via `calibration.n_cal` and
`calibration.n_test` in your `trustgate.yaml`. If you have 1000 questions, a
500/500 split is a good default.

---

## 3. How much does it cost?

Cost depends on three factors: the number of questions, the number of samples
per question (K), and the model's pricing.

**Rough estimates for 500 questions at K=10:**

| Model | Approximate cost |
|---|---|
| GPT-4.1-mini | ~$5-20 |
| GPT-4.1 | ~$50-100 |
| GPT-4.1-nano | ~$1-5 |
| Claude Haiku 3.5 | ~$10-30 |

Costs are dominated by output tokens. Shorter answers (math, MCQ) are cheaper
than long-form generation.

Re-runs are free thanks to response caching (see question 7).

---

## 4. Can I use this with Anthropic, Together, or other providers?

Yes. TrustGate works with any OpenAI-compatible API endpoint. This includes:

- **OpenAI** (GPT-4.1, GPT-4o, etc.)
- **Anthropic** (Claude models, via native API support)
- **Together AI** (open-source models)
- **vLLM**, **Ollama**, **LiteLLM**, and any other OpenAI-compatible server

Configure the endpoint in `trustgate.yaml`:

```yaml
endpoint:
  url: "https://api.anthropic.com/v1/messages"
  model: "claude-sonnet-4-6"
  api_key_env: "ANTHROPIC_API_KEY"
  provider: "anthropic"
```

The `provider` field is auto-detected from the URL if omitted. Set it
explicitly for non-standard endpoints.

---

## 5. What if I don't have ground truth labels?

You have several options:

1. **Use questions with known answers.** Set the `acceptable_answers` field on
   each `Question` object. The built-in dataset loaders (GSM8K, MMLU,
   TruthfulQA) do this automatically.

2. **Provide a labels file.** Pass a CSV or JSON file via the
   `ground_truth_file` parameter:
   - CSV format: columns `id` and `label` with a header row.
   - JSON format: `{"question_id": "correct_answer", ...}`

3. **Use the human calibration UI.** TrustGate supports manual labeling
   workflows where a human reviews model responses and marks them as
   correct or incorrect. This is particularly useful for open-ended tasks
   where automated evaluation is difficult.

---

## 6. Why is my reliability level low?

A low reliability level (e.g., below 0.80) can have several causes:

- **The model is genuinely underperforming on this task.** Check the
  `capability_gap` metric -- if it is high, the model cannot even produce the
  correct answer in K attempts for many questions.

- **Too few questions.** With fewer than 100 questions, conformal calibration
  may not have enough statistical power. Try increasing to 500+.

- **Wrong canonicalizer.** If the canonicalizer does not correctly map equivalent
  answers to the same canonical form, self-consistency scores will be
  artificially low. For example, using `"mcq"` on math problems instead of
  `"numeric"` will produce poor results.

- **K is too low.** With very few samples per question (e.g., K=3), the
  self-consistency profile is noisy. Try increasing `sampling.k_fixed` to 10 or
  higher.

- **Temperature is too low.** A temperature near 0 produces near-identical
  samples, which defeats the purpose of self-consistency. The default of 0.7 is
  usually a good choice.

---

## 7. How does caching work?

All API responses are cached on disk in `~/.trustgate/cache/`. The cache key
is derived from the provider, model name, prompt text, temperature, and sample
index.

**Key behaviors:**

- **First run:** All questions are sent to the API. Responses are saved to the
  cache.
- **Re-runs:** Cached responses are loaded instantly. No API calls are made.
  Cost is zero.
- **Changing parameters:** If you change the model, temperature, or question
  text, new API calls are made (different cache key). Previous cached responses
  are not deleted.

To clear the cache, delete the `~/.trustgate/cache/` directory.

---

## 8. Can I use this in CI/CD?

Yes. The `trustgate` CLI supports a `--min-reliability` flag for automated
pass/fail gating:

```bash
trustgate certify --config trustgate.yaml --min-reliability 0.90
```

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | PASS -- reliability level meets or exceeds the threshold. |
| 1 | FAIL -- reliability level is below the threshold. |

This integrates naturally with CI systems like GitHub Actions, GitLab CI, and
Jenkins. A typical workflow:

```yaml
# .github/workflows/trustgate.yml
- name: Run TrustGate certification
  run: trustgate certify --config trustgate.yaml --min-reliability 0.90
  env:
    LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
```

You can also export results to JSON or CSV for artifact storage:

```python
from theaios.trustgate.reporting.json_export import export_json

result = trustgate.certify(config_path="trustgate.yaml")
export_json(result, path="trustgate-report.json")
```

---

## 9. What Python versions are supported?

TrustGate requires **Python 3.10 or later**.

This is due to the use of modern Python features including `X | Y` union type
syntax, `match` statements, and `dataclasses` with enhanced field support.

---

## 10. How do I contribute?

See the [repository](https://github.com/Cohorte-ai/trustgate) for full guidelines. In brief:

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Make your changes with tests.
4. Run the test suite: `pytest`.
5. Run linting: `ruff check src/ tests/`.
6. Open a pull request against `main`.

Contributions are welcome for new canonicalizers, dataset loaders, reporting
formats, and documentation improvements.
