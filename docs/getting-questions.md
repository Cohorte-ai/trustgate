# Getting Your Questions

The most common pushback: *"I can't use TrustGate because I don't have a dataset of questions."*

You don't need one upfront. You don't need a gold-standard benchmark. You don't even need correct answers — that's what [human calibration](human-calibration.md) is for. You just need **questions that represent what your system will face in production**.

Here are four ways to get them.

---

## 1. Generate questions with AI

The fastest path. Ask any LLM to generate realistic questions for your use case.

### Prompt template

```
You are helping me test an AI system. The system is: [DESCRIBE YOUR SYSTEM].

Generate 100 realistic questions that real users would ask this system.
The questions should:
- Cover the full range of topics the system handles
- Include easy questions (the system should always get right)
- Include hard/ambiguous questions (where the system might struggle)
- Include edge cases and unusual phrasings
- Be representative of actual production traffic

Output as CSV with columns: id, question
```

### Example: e-commerce chatbot

```
Generate 100 realistic customer support questions for an e-commerce chatbot
that handles orders, returns, shipping, product info, and account issues.
Include common questions, edge cases, and questions in informal language.
```

This gives you questions like:
```csv
id,question
q001,"Where is my order #12345?"
q002,"Can I return a product after 30 days?"
q003,"Do you ship to Canada?"
q004,"The item I received is damaged, what do I do?"
q005,"hey can u change my shipping address its wrong lol"
```

### Tips

- **Generate more than you need.** Generate 200, use 100 for calibration. You can always use the rest for future recalibration.
- **Review the list.** Remove duplicates, add questions the AI missed, and ensure coverage of important edge cases.
- **Use your domain knowledge.** You know the hard cases better than the AI does. Add the questions that keep you up at night.
- **Vary the phrasing.** Real users don't write in perfect English. Include informal, misspelled, or ambiguous questions.

---

## 2. Extract from production logs

If your system is already in production (even in beta), your **observability data is the best source of questions**. Real users generate the most representative test set possible.

### Where to find them

| Observability tool | How to extract |
|-------------------|----------------|
| **Langfuse** | Export traces → extract `input` field |
| **LangSmith** | Datasets tab or export run inputs |
| **Datadog LLM Observability** | Export traces as CSV |
| **Custom logging** | Query your database for user prompts |
| **Intercom / Zendesk** | Export conversation starters |
| **Application logs** | Grep for the input field in your API logs |

### Example: extracting from logs

```python
import json

# Your logs might look like this
logs = [
    {"timestamp": "...", "user_input": "What is your return policy?", "response": "..."},
    {"timestamp": "...", "user_input": "How do I reset my password?", "response": "..."},
    # ...
]

# Convert to TrustGate questions format
questions = [
    {"id": f"prod_{i}", "question": log["user_input"]}
    for i, log in enumerate(logs)
]

with open("questions.json", "w") as f:
    json.dump(questions, f, indent=2)
```

### Tips

- **Sample randomly.** Don't cherry-pick — random sampling gives an unbiased test set.
- **Deduplicate.** Many users ask the same thing. Keep one representative version of each.
- **Include the hard ones.** Don't filter out the questions your system struggled with — those are the most valuable for calibration.
- **Respect privacy.** Remove PII (names, emails, account numbers) before using production data for certification. Replace with placeholders.

---

## 3. Augment with AI

Start with a small set (from logs or manual creation) and expand it with AI:

```
Here are 20 real questions our system receives:
[paste your 20 questions]

Generate 80 more questions in the same style, covering topics and phrasings
that are not represented above. Maintain the same level of formality and
the same types of requests.
```

This hybrid approach gives you the realism of production data with the coverage of AI generation.

---

## 4. Use built-in benchmarks

For standard tasks (math, knowledge, reasoning), TrustGate ships dataset loaders:

```python
from theaios.trustgate.datasets import load_gsm8k, load_mmlu, load_truthfulqa

# Grade-school math (GSM8K)
questions = load_gsm8k(n=200)

# Multiple-choice knowledge (MMLU)
questions = load_mmlu(subjects=["abstract_algebra", "anatomy"], n=100)

# Truthfulness (TruthfulQA)
questions = load_truthfulqa(n=100)
```

These come with ground truth labels, so you can skip human calibration entirely.

---

## What about ground truth labels?

You have three options:

### Option A: You have labels (easiest)

If your questions come with known correct answers — from a benchmark, a test set, or manual annotation — put them in the `acceptable_answers` column:

```csv
id,question,acceptable_answers
q001,"What is 2+2?","4"
q002,"Capital of France?","Paris"
```

TrustGate uses these directly. No human calibration needed.

### Option B: No labels — use human calibration

This is the common case for production systems. You have questions but no correct answers.

1. Run `trustgate calibrate --export questionnaire.html`
2. Share the HTML file with a domain expert
3. They pick the acceptable answer for each question (10 minutes for 50 items)
4. Run `trustgate certify --ground-truth labels.json`

See [Human Calibration](human-calibration.md) for the full guide.

### Option C: No labels, no human — use auto-judge

If you can't get a human reviewer, use an LLM to automatically pick the
correct answer from the ranked canonical answers (automated calibration):

```bash
trustgate certify --auto-judge --config trustgate.yaml
```

This requires a `judge_endpoint` in your config (the LLM that judges):

```yaml
canonicalization:
  type: "llm"       # or mcq, numeric, etc.
  judge_endpoint:    # also used by --auto-judge for calibration
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1-nano"
    api_key_env: "LLM_API_KEY"
```

The auto-judge replaces the human in the calibration step — it looks at
each question and its ranked canonical answers, and picks the correct one.
This is less rigorous than human calibration (the judge has irreducible
bias — see the [paper](https://arxiv.org/abs/2602.21368), Proposition 3.4),
but it's fully automated.

---

## How many questions do I need?

| Calibration set size | Reliability estimate quality |
|---------------------|------------------------------|
| 30 | Rough estimate (large confidence interval) |
| 50 | Reasonable for initial assessment |
| 100 | Good for most production use cases |
| 250+ | High-precision certification |
| 500+ | Publication-grade results |

The conformal coverage gap is bounded by 1/(n+1), where n is the calibration set size. With 50 items, the gap is at most 2%. With 500 items, it's at most 0.2%.

You also need a **test set** of equal size (TrustGate splits automatically). So for 100 calibration items, prepare at least 200 questions total.

---

## Putting it all together

```bash
# 1. Generate or extract questions
#    → questions.csv (at least 200 rows)

# 2. Certify (with ground truth labels)
trustgate certify --questions questions.csv

# 2. Or certify (with human calibration)
trustgate calibrate --export questionnaire.html --questions questions.csv
# → share with expert → get labels.json back
trustgate certify --questions questions.csv --ground-truth labels.json
```

The hardest part is getting started. Once you have 50-100 representative questions, TrustGate handles the rest.
