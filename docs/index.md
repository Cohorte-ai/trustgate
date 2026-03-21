# TrustGate

**Know if your AI is ready to ship -- one number, one guarantee.**

TrustGate certifies the reliability of any LLM or AI endpoint using self-consistency sampling and conformal prediction. Point it at your API, run the certification pipeline, and get a single reliability level (e.g., 94.6%) backed by a formal statistical guarantee -- not a vibe, not a leaderboard score, a mathematical proof. It works with any provider, any task type, any model, entirely black-box with no model internals required.

---

## Installation

```bash
pip install theaios-trustgate
```

Optional extras for specific features:

```bash
# LLM-as-judge canonicalization (needs openai SDK)
pip install "theaios-trustgate[judge]"

# Embedding-based canonicalization (needs sentence-transformers + hdbscan)
pip install "theaios-trustgate[embedding]"

# Local human calibration web UI (needs Flask)
pip install "theaios-trustgate[serve]"

# Everything
pip install "theaios-trustgate[all]"
```

Requires Python 3.10 or later.

---

## Quickstart

### Step 1: Create a config file

Create a file called `trustgate.yaml` in your project directory:

```yaml
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"

sampling:
  k_fixed: 10
  sequential_stopping: true

canonicalization:
  type: "mcq"

calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.15, 0.20]
  n_cal: 250
  n_test: 250

questions:
  file: "questions.csv"
```

Your `questions.csv` should look like this:

```csv
id,question,acceptable_answers
q001,"What is the capital of France? (A) London (B) Paris (C) Berlin (D) Madrid","B"
q002,"Which planet is largest? (A) Earth (B) Mars (C) Jupiter (D) Venus","C"
```

### Step 2: Run certification

```bash
trustgate certify
```

### Step 3: Read your reliability level

TrustGate prints a summary like this:

```
TrustGate Certification Result
--------------------------------------------
  Reliability Level:   94.6%  (CI: 93.2-95.8%)
  M* (prediction set): 1
  Empirical Coverage:   0.956  (target: 0.900)
  Conditional Coverage: 0.980
  Capability Gap:       2.4%
  Items:                250 cal / 250 test
  Sampling:             K=10, saved $11.20 (47%) via sequential stopping
  Status:               PASS
--------------------------------------------
```

The **reliability level** (94.6% in this example) is the single number that matters. It tells you: for at least 94.6% of the questions you tested, the model's top answer was correct -- and this claim holds with a formal statistical guarantee. If you are evaluating whether an AI is reliable enough to deploy, this is the number to look at. Unlike raw accuracy or leaderboard rankings, the reliability level is backed by conformal prediction, which means it accounts for statistical uncertainty and does not overfit to the particular test set you used.

---

## What do the other metrics mean?

- **M\* (prediction set size)**: How many top answers you need to include to guarantee the correct answer is covered. M\*=1 means the single most frequent answer is enough. M\*=2 means you need the top two.
- **Empirical coverage**: The fraction of test questions where the top-M\* answers included the correct one.
- **Conditional coverage**: Same as empirical coverage, but computed only on questions the model can actually answer (where the correct answer appeared at least once across K samples).
- **Capability gap**: The fraction of questions where the correct answer never appeared in any of the K samples. These are questions the model fundamentally cannot answer.

For a deeper explanation of the math behind these metrics, see [Concepts](concepts.md).

---

## Further reading

- [Concepts](concepts.md) -- Plain-language explanation of the statistical methods (self-consistency, conformal prediction, sequential stopping)
- [Configuration](configuration.md) -- Full reference for `trustgate.yaml`
- [CLI Reference](cli.md) -- All commands and flags for the `trustgate` CLI
- [Canonicalization](canonicalization.md) -- How raw LLM responses are normalized, and how to write your own canonicalizer
- [Human Calibration](human-calibration.md) -- Using human reviewers when you don't have ground-truth labels
- [API Reference](api-reference.md) -- Python API for programmatic use
- [FAQ](faq.md) -- Common questions and troubleshooting
