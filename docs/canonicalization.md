# Canonicalization

Canonicalization is the process of converting raw, free-text LLM responses into
short, comparable canonical forms. Two answers that mean the same thing should
produce the same canonical string. TrustGate uses these canonical forms to
measure self-consistency across repeated samples.

---

## Built-in Canonicalizers

TrustGate ships with five built-in canonicalizers. Select one with
`canonicalization.type` in your `trustgate.yaml` or via `--task-type` on the CLI.

### `mcq` --- Multiple-Choice

Extracts the chosen option letter (A through E) from a free-text response. Uses
a priority chain of regex patterns: explicit "the answer is (B)" phrases, a
leading letter, a trailing letter, and finally a standalone letter if it is the
only candidate.

| Input                          | Output |
|--------------------------------|--------|
| `"I think B) Paris"`           | `"B"`  |
| `"The correct answer is (C)."` | `"C"`  |
| `"A"`                          | `"A"`  |
| `"(no clear letter)"`          | `""`   |

**Config:**

```yaml
canonicalization:
  type: "mcq"
```

**Source:** `trustgate.canonicalize.mcq.MCQCanonicalizer`

---

### `numeric` --- Numeric Extraction

Extracts and normalizes the final number from a response. Handles currency
symbols, commas, percentages, fractions, LaTeX `\boxed{}` notation, and the
GSM8K `####` delimiter. Falls back to the last number found in the text.

| Input                                 | Output   |
|---------------------------------------|----------|
| `"The answer is $42.50"`              | `"42.5"` |
| `"#### 7"`                            | `"7"`    |
| `"\boxed{3/4}"`                       | `"0.75"` |
| `"About 15% of the population"`       | `"0.15"` |
| `"I calculated 1,200 widgets total."` | `"1200"` |

**Config:**

```yaml
canonicalization:
  type: "numeric"
```

**Source:** `trustgate.canonicalize.numeric.NumericCanonicalizer`

---

### `llm_judge` --- LLM-as-Judge

Sends the (question, answer) pair to a separate judge LLM that returns
`"correct"` or `"incorrect"`. The judge receives a structured prompt asking it
to evaluate correctness and reply with exactly one word. Includes automatic
retries with exponential backoff.

Requires a `judge_endpoint` configuration pointing to the judge model.

**Config:**

```yaml
canonicalization:
  type: "llm_judge"
  judge_endpoint:
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1"
    api_key_env: "LLM_API_KEY"
```

**Source:** `trustgate.canonicalize.llm_judge.LLMJudgeCanonicalizer`

---

### `llm` --- LLM Semantic Canonicalization

Uses a lightweight LLM to extract the core factual answer from a free-text
response, producing a short canonical form. This is a true canonicalizer
(Definition 4.1) — it groups semantically equivalent answers without judging
correctness. Correctness is determined later, during calibration (by a human
or by `--auto-judge`).

| Input | Output |
|-------|--------|
| `"The capital of France is Paris"` | `"paris"` |
| `"I believe it's Paris"` | `"paris"` |
| `"London"` | `"london"` |
| `"The answer is approximately 42.5"` | `"42.5"` |

Use a cheap, fast model (e.g., `gpt-4.1-nano`) — it only extracts a short
string.

**Config:**

```yaml
canonicalization:
  type: "llm"
  judge_endpoint:
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1-nano"
    api_key_env: "LLM_API_KEY"
```

**Source:** `trustgate.canonicalize.llm_semantic.LLMSemanticCanonicalizer`

---

### `embedding` --- Semantic Clustering

Clusters responses by semantic similarity using sentence-transformers embeddings
and HDBSCAN clustering. Each answer is assigned a label like `"cluster_0"`,
`"cluster_1"`, etc. Noise points that do not fit any cluster receive unique
singleton labels.

This canonicalizer exposes a `canonicalize_batch` method that processes all
answers for a given question at once (required for clustering). The single-answer
`canonicalize` method returns the preprocessed text as a fallback.

The default embedding model is `all-MiniLM-L6-v2`. If HDBSCAN is not installed,
a built-in greedy cosine-similarity clustering fallback is used.

Requires the embedding extra:

```bash
pip install "theaios-trustgate[embedding]"
```

**Config:**

```yaml
canonicalization:
  type: "embedding"
```

**Source:** `trustgate.canonicalize.embedding.EmbeddingCanonicalizer`

---

## Shared Preprocessing

Every canonicalizer inherits from the `Canonicalizer` base class, which provides
a `preprocess` method applied before canonicalization. The preprocessing steps
are:

1. Strip leading and trailing whitespace
2. Remove markdown code fences (keeps the content inside them)
3. Remove common LLM preambles ("Sure!", "Certainly,", "I think", "Here is the
   answer:", etc.)
4. Normalize unicode to NFC form

Canonicalizers call `self.preprocess(answer)` at the start of their
`canonicalize` method. You should do the same in custom canonicalizers.

---

## Plugin System

You can write your own canonicalizer for domain-specific tasks. The plugin
system has three steps:

1. **Inherit** from `trustgate.Canonicalizer`
2. **Implement** `canonicalize(self, question: str, answer: str) -> str`
3. **Register** with the `@trustgate.register_canonicalizer("my_name")` decorator

Once registered, your canonicalizer can be used with
`canonicalization.type: "my_name"` in the config or `--task-type my_name` on the
CLI.

### Registration via Decorator

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer

@register_canonicalizer("my_name")
class MyCanonicalizer(Canonicalizer):
    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        # Your logic here
        return text
```

The decorator adds your class to the global registry. When TrustGate encounters
`type: "my_name"` in the config, it instantiates your class via
`get_canonicalizer("my_name")`.

### Registration via Dotted Path (No Decorator)

Alternatively, use the `custom` type with `custom_class` pointing to the fully
qualified class name:

```yaml
canonicalization:
  type: "custom"
  custom_class: "my_package.my_module.MyCanonicalizer"
```

This uses `importlib` to dynamically import and instantiate the class. The class
must still inherit from `Canonicalizer`.

### Optional Methods

Beyond the required `canonicalize` method, you can optionally override:

- **`validate(self, canonical: str) -> bool`** --- Check that a canonical form
  is well-formed. Called after canonicalization for sanity checking. Returns
  `True` by default.

- **`canonicalize_batch(self, question: str, answers: list[str]) -> list[str]`**
  --- Process a batch of answers at once. Useful when canonicalization benefits
  from seeing all answers together (like the `embedding` canonicalizer). If not
  implemented, TrustGate falls back to calling `canonicalize` on each answer
  individually.

### Constructor Arguments

Your canonicalizer can accept keyword arguments in `__init__`. These are passed
through when the canonicalizer is instantiated:

```python
@register_canonicalizer("threshold_numeric")
class ThresholdNumericCanonicalizer(Canonicalizer):
    def __init__(self, threshold: float = 0.5, **kwargs: object) -> None:
        self.threshold = threshold

    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        try:
            value = float(text)
            return "above" if value >= self.threshold else "below"
        except ValueError:
            return "invalid"
```

---

## Full Custom Canonicalizer Example

Below is a complete example of a custom canonicalizer for medical diagnosis
tasks. It normalizes ICD-10 codes extracted from free-text LLM responses:

```python
"""Custom canonicalizer for medical diagnosis (ICD-10 code extraction)."""

import re
from theaios.trustgate import Canonicalizer, register_canonicalizer

# ICD-10 codes look like: A00-Z99 with optional decimal (e.g., J18.9, E11.65)
_ICD10_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")


@register_canonicalizer("icd10")
class ICD10Canonicalizer(Canonicalizer):
    """Extract and normalize ICD-10 diagnosis codes from LLM responses."""

    def __init__(self, primary_only: bool = True, **kwargs: object) -> None:
        self.primary_only = primary_only

    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        if not text:
            return "no_code"

        codes = _ICD10_RE.findall(text.upper())
        if not codes:
            return "no_code"

        if self.primary_only:
            return codes[0]

        # Return all unique codes sorted, pipe-delimited
        return "|".join(sorted(set(codes)))

    def validate(self, canonical: str) -> bool:
        if canonical == "no_code":
            return True
        # Each code (or pipe-delimited codes) should match ICD-10 format
        parts = canonical.split("|")
        return all(_ICD10_RE.fullmatch(p) for p in parts)
```

**Usage in config:**

```yaml
canonicalization:
  type: "icd10"
```

**Usage in Python:**

```python
from theaios import trustgate

# Make sure the module containing the canonicalizer is imported first
import my_medical_canonicalizers  # noqa: F401 (triggers @register_canonicalizer)

result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="LLM_API_KEY",
        ),
        canonicalization=trustgate.CanonConfig(type="icd10"),
    ),
    questions=my_questions,
    labels=my_labels,
)
```

---

## Listing Available Canonicalizers

To see all registered canonicalizers (built-in and custom):

```python
from theaios.trustgate.canonicalize import list_canonicalizers

print(list_canonicalizers())
# ['embedding', 'llm', 'llm_judge', 'mcq', 'numeric']
```

---

## Choosing the Right Canonicalizer

| Task type                       | Recommended canonicalizer | Notes                                          |
|---------------------------------|---------------------------|-------------------------------------------------|
| Multiple-choice exams           | `mcq`                     | Works out of the box for A/B/C/D/E questions    |
| Math / quantitative             | `numeric`                 | Handles currency, fractions, LaTeX              |
| Code generation                 | Custom plugin             | Write a custom canonicalizer with your own sandboxing (see security note below) |
| Open-ended / free-text          | `llm`                     | Semantic grouping via LLM; lightweight           |
| Binary correct/incorrect        | `llm_judge`               | Paper Section 4.3 regime (3); coarse but fast   |
| Free-text with semantic overlap | `embedding`               | Groups similar answers; no ground truth needed   |
| Domain-specific                 | Custom plugin             | Write your own for maximum control              |

> **Security note:** The `code_exec` canonicalizer was removed from the core
> package because it executed untrusted code generated by LLMs. Executing
> arbitrary code --- even in a subprocess with a restricted environment --- is
> inherently dangerous and should not be a default option in a library. If you
> need code execution canonicalization, implement it as a custom canonicalizer
> plugin with your own sandboxing, containerization, and security controls
> appropriate to your environment.

---

## Certifying Pipeline Components

Complex AI systems are multi-step pipelines. Instead of certifying the final
output (which is often long and hard to canonicalize), **certify each component
independently**. This lets you pinpoint exactly where reliability breaks down
and iterate on the weak link without re-certifying the whole system.

```
Query → [Retriever] → [Reranker] → [Generator] → Answer
            ↑              ↑             ↑
      certify: 94%    certify: 91%   certify: 87%
```

Each component is just an endpoint — give it a question, get a short structured
output. TrustGate certifies it independently with its own questions and
canonicalization.

| Pipeline component | What it outputs | Canonicalization |
|--------------------|-----------------|------------------|
| RAG retriever | Retrieved document IDs | Exact match (custom) |
| SQL agent | SQL query | Normalized SQL (custom) |
| Classification step | Category label | `mcq` |
| Entity extraction | Entity list | Sorted list (custom) |
| Reasoning / chain-of-thought | Intermediate conclusion | `llm` or custom |
| Final short answer | Structured value | `numeric` or `mcq` |

### Why component-level certification matters

- **Pinpoint failures.** "The generator is the weak link, not the retriever."
- **Iterate faster.** Improve one component, re-certify just that one — not
  the full pipeline.
- **Stay agnostic to data changes.** If your RAG corpus changes, re-certify
  the retriever. The generator certification is still valid.
- **Quantify cost/reliability tradeoffs per component.** A cheap model might
  be reliable enough for retrieval but not for reasoning.

### Example: certifying a RAG retriever

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer

@register_canonicalizer("retriever_docs")
class RetrieverCanonicalizer(Canonicalizer):
    def canonicalize(self, question: str, answer: str) -> str:
        """Extract sorted document IDs from retriever output."""
        import json
        try:
            docs = json.loads(answer)
            doc_ids = sorted(d["id"] for d in docs)
            return "|".join(doc_ids)
        except (json.JSONDecodeError, KeyError):
            return ""
```

```yaml
# trustgate-retriever.yaml
endpoint:
  url: "https://my-rag.example.com/api/retrieve"
  temperature: null
  request_template:
    query: "{{question}}"
  response_path: "documents"
  cost_per_request: 0.001

canonicalization:
  type: "retriever_docs"
```

```bash
trustgate certify --config trustgate-retriever.yaml
```

Now you know the retriever's reliability independently of the generator.
