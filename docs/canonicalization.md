# Canonicalization

Canonicalization is the process of converting raw, free-text LLM responses into
short, comparable canonical forms. Two answers that mean the same thing should
produce the same canonical string. TrustGate uses these canonical forms to
measure self-consistency across repeated samples.

---

## Built-in Canonicalizers

TrustGate ships with five canonicalizers. Select one with `canonicalization.type`
in your `trustgate.yaml` or via `--task-type` on the CLI.

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

### `code_exec` --- Code Execution

Extracts a Python code block from the response (or treats the entire response as
code), writes it to a temporary file, and runs it in a sandboxed subprocess with
a restricted environment (no network, no home directory access). Returns `"pass"`
if the script exits with code 0, or `"fail"` otherwise (including timeouts).

| Input                                | Output   |
|--------------------------------------|----------|
| `` ```python\nprint("hello")\n``` `` | `"pass"` |
| `"import sys; sys.exit(1)"`          | `"fail"` |
| (infinite loop, exceeds timeout)     | `"fail"` |

The default timeout is 10 seconds per execution.

**Config:**

```yaml
canonicalization:
  type: "code_exec"
```

**Source:** `trustgate.canonicalize.code_exec.CodeExecCanonicalizer`

---

### `llm_judge` --- LLM-as-Judge

Sends the (question, answer) pair to a separate judge LLM that returns
`"correct"` or `"incorrect"`. The judge receives a structured prompt asking it
to evaluate correctness and reply with exactly one word. Includes automatic
retries with exponential backoff.

Requires a `judge_endpoint` configuration pointing to the judge model. Install
the judge extra for OpenAI SDK support:

```bash
pip install "trustgate[judge]"
```

**Config:**

```yaml
canonicalization:
  type: "llm_judge"
  judge_endpoint:
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1"
    api_key_env: "OPENAI_API_KEY"
```

**Source:** `trustgate.canonicalize.llm_judge.LLMJudgeCanonicalizer`

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
pip install "trustgate[embedding]"
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
from trustgate import Canonicalizer, register_canonicalizer

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
from trustgate import Canonicalizer, register_canonicalizer

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
import trustgate

# Make sure the module containing the canonicalizer is imported first
import my_medical_canonicalizers  # noqa: F401 (triggers @register_canonicalizer)

result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
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
from trustgate.canonicalize import list_canonicalizers

print(list_canonicalizers())
# ['code_exec', 'embedding', 'llm_judge', 'mcq', 'numeric']
```

---

## Choosing the Right Canonicalizer

| Task type                       | Recommended canonicalizer | Notes                                          |
|---------------------------------|---------------------------|-------------------------------------------------|
| Multiple-choice exams           | `mcq`                     | Works out of the box for A/B/C/D/E questions    |
| Math / quantitative             | `numeric`                 | Handles currency, fractions, LaTeX              |
| Code generation                 | `code_exec`               | Tests if generated code runs without errors     |
| Open-ended / subjective         | `llm_judge`               | Needs a judge endpoint; adds API cost           |
| Free-text with semantic overlap | `embedding`               | Groups similar answers; no ground truth needed   |
| Domain-specific                 | Custom plugin             | Write your own for maximum control              |
