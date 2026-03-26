# Concepts

This page explains the statistical methods behind TrustGate in plain language. You do not need a statistics background to use TrustGate, but understanding these ideas will help you interpret results and tune parameters.

---

## 1. Self-Consistency Sampling

**The idea:** Ask the same question K times with temperature > 0. If the AI gives the same answer 9 out of 10 times, it is probably right. If it gives a different answer every time, it is guessing.

When you call an LLM with temperature > 0, you get stochastic (randomly varying) outputs. TrustGate exploits this: by asking the same question multiple times, the pattern of agreement across responses reveals how confident the model actually is. A question where the model consistently produces the same answer is one it "knows." A question where answers are scattered is one it is uncertain about.

TrustGate collects K responses per question, then counts how often each distinct answer appears. This produces a **self-consistency profile** -- a ranked list of (answer, frequency) pairs. For example:

```
Question: "What is the capital of France?"
  "Paris"  -> 9/10  (90%)
  "Lyon"   -> 1/10  (10%)
```

The profile above tells us the model is highly consistent on this question. The top answer ("Paris") appeared 90% of the time.

Before counting, raw LLM outputs are passed through a **canonicalizer** that normalizes them into comparable forms (e.g., extracting the option letter from "I think the answer is B) Paris" to produce just `"B"`). This ensures that superficially different phrasings of the same answer are grouped together.

---

## 2. Conformal Prediction

**The idea:** A statistical method that converts raw consistency patterns into a guaranteed reliability number. Unlike heuristic confidence scores, conformal prediction provides mathematically provable coverage guarantees.

Here is how it works at a high level:

1. **Calibration set:** Take a set of questions where you know the correct answers. For each question, look at the self-consistency profile and find the position of the correct answer. If the correct answer is the most frequent response, its position is 1. If it is the second most frequent, its position is 2. This position is the **nonconformity score** -- it measures how "surprising" the correct answer's placement is.

2. **Quantile threshold:** Sort all the calibration nonconformity scores and pick a threshold (called M\*) at a chosen confidence level. This threshold tells you: "If you include the top M\* answers, you will cover the correct answer for at least (1 - alpha) of questions."

3. **Test set:** Apply this threshold to a held-out test set and verify that coverage actually holds.

The key property of conformal prediction is that the coverage guarantee is **distribution-free** -- it holds regardless of the underlying data distribution, with no assumptions about how the model generates answers. If TrustGate reports a reliability level of 98.0%, that means the model's top answer is correct for at least 98.0% of questions, with a formal guarantee.

### Where do the correct answers come from?

There are two ways to provide the correct answers for the calibration set:

**Option A: Ground truth labels.** If you have a dataset with known answers (e.g., a benchmark like GSM8K or MMLU), you supply them directly. TrustGate looks up each correct answer in the self-consistency profile and computes the nonconformity score automatically.

**Option B: Human calibration.** When you don't have ground truth, a human reviewer provides the answers. For each question, TrustGate shows all the candidate answers the AI produced (in randomized order, with no ranking or frequency information). The reviewer picks the acceptable one. The system then resolves which rank that answer held in the original self-consistency profile — that rank is the nonconformity score.

The two options are mathematically equivalent: both produce a nonconformity score per question, which feeds into the same conformal calibration procedure. The human is not grading or scoring — they are identifying which answer is correct so the system can measure where it falls in the AI's own ranking.

### Worked example: from human clicks to reliability level

Suppose you have 50 calibration questions and a human reviewer labels each one:

| Question | AI's ranked answers | Human picks | Nonconformity score |
|----------|-----------------------|-------------|---------------------|
| q1 | "Paris" (8/10), "Lyon" (2/10) | "Paris" | 1 (rank 1) |
| q2 | "42" (6/10), "43" (3/10), "44" (1/10) | "42" | 1 (rank 1) |
| q3 | "B" (5/10), "C" (3/10), "A" (2/10) | "C" | 2 (rank 2) |
| q4 | "Aspirin" (4/10), "Ibuprofen" (4/10), "Tylenol" (2/10) | "Ibuprofen" | 2 (rank 2) |
| q5 | "Red" (5/10), "Blue" (3/10), "Green" (2/10) | (none) | ∞ (not in list) |

The human doesn't see the frequencies or ranks — they see the answers in random order and judge on content alone. TrustGate resolves the rank internally.

Now we have 50 scores: mostly 1s (the AI's top answer was correct), some 2s (correct answer was second), and a few ∞s (correct answer never appeared — capability gap). These scores are sorted, and the conformal quantile at the desired alpha level gives M\*. If most scores are 1, then M\*=1 and the reliability level is high.

---

## 3. Reliability Level

**The idea:** The reliability level is the highest confidence at which the model's top answer (mode) is trustworthy. It is the single headline number TrustGate produces.

The formula (Definition 2.4 in the paper):

```
reliability level = |{calibration items where mode is correct}| / (n + 1)
```

For example, if 49 out of 50 calibration items have the correct answer as the mode:

```
reliability = 49 / (49 + 1) = 98.0%
```

This means: "the model's top answer is correct for 98.0% of questions, with a formal guarantee." The `n+1` denominator (instead of `n`) is the conformal correction that ensures the guarantee is conservative.

**Understanding alpha (for M\*):** The `--alpha` parameter controls the confidence level for the prediction set size M\*. Alpha = 0.05 means "at 95% confidence, how many top answers do I need?" A smaller alpha demands stricter guarantees, so M\* may be larger.

---

## 4. M* (Prediction Set Size)

**The idea:** M\* is how many top answers you need to include in your prediction set to guarantee the correct answer is in there. M\*=1 is ideal -- it means the single most popular answer is enough.

Think of it like a multiple-choice guarantee. If M\*=1, TrustGate is saying: "The model's top answer is correct often enough to meet the coverage target." If M\*=2, it is saying: "You need to consider the model's top two answers to get that same guarantee."

In practice:

- **M\*=1** means the model is decisive and accurate. The single most consistent answer is correct for enough questions to meet the reliability target.
- **M\*=2** means the model is sometimes torn between two plausible answers. You still get the guarantee, but you need to present two candidate answers instead of one.
- **M\*=3+** means the model is frequently uncertain. The guarantee holds, but the prediction set is large, which is less useful in practice.

M\* is computed from the conformal quantile of the nonconformity scores on the calibration set at the user's target alpha (default 0.05 = 95% confidence). Use `--alpha` to change the confidence level:

```bash
trustgate certify --alpha 0.01   # 99% confidence → M* may be larger
trustgate certify --alpha 0.05   # 95% confidence (default)
trustgate certify --alpha 0.10   # 90% confidence → M* may be smaller
```

---

## 5. Capability Gap

**The idea:** The capability gap is the fraction of questions the AI simply cannot answer, even with multiple attempts. These are questions where the correct answer never appeared in any of the K samples.

If you ask a model "What is the 47th digit of pi?" ten times and it never produces the correct answer in any sample, that question falls in the capability gap. No amount of sampling or statistical analysis can help -- the model does not have this knowledge.

The capability gap is computed as:

```
capability_gap = (number of questions where correct answer never appeared) / (total questions)
```

A capability gap of 2.4% means the model is fundamentally unable to answer 2.4% of the test questions. This is a hard floor on reliability -- even a perfect statistical method cannot push reliability above (1 - capability_gap).

The capability gap is useful for distinguishing between two failure modes:
- **The model knows the answer but is inconsistent** (addressable by increasing K or improving canonicalization)
- **The model does not know the answer at all** (a fundamental limitation of the model)

---

## 6. Coverage

TrustGate reports two types of coverage:

### Empirical Coverage

The fraction of **all** test questions where the top-M\* answers include the correct answer. This is the standard coverage metric and the one used to determine the reliability level.

```
empirical_coverage = (questions where correct answer is in top M*) / (all test questions)
```

### Conditional Coverage

The fraction of **solvable** test questions where the top-M\* answers include the correct answer. A question is "solvable" if the correct answer appeared at least once across the K samples.

```
conditional_coverage = (questions where correct answer is in top M*) / (solvable questions only)
```

Conditional coverage is always greater than or equal to empirical coverage. The gap between them reflects the capability gap: questions the model cannot answer at all drag down empirical coverage but do not affect conditional coverage.

Conditional coverage is useful for understanding how well the model performs *when it can perform*. If conditional coverage is high (e.g., 98%) but empirical coverage is lower (e.g., 94%), the remaining errors are almost entirely due to the capability gap rather than inconsistency.

---

## 7. Sequential Stopping

**The idea:** You do not always need all K samples. TrustGate uses Hoeffding bounds to detect when the answer pattern has stabilized and stops early, typically saving around 50% of API costs.

Without sequential stopping, TrustGate sends K requests per question regardless of how clear the answer pattern is. For a question where the model says "Paris" on the first 5 attempts, there is little value in the remaining 5 requests.

Sequential stopping works as follows:

1. Sample responses one at a time (or in small batches).
2. After each new sample, compute the mode frequency (how often the most common answer has appeared so far).
3. Apply a **Hoeffding bound** to determine whether the mode is statistically dominant. Specifically, check whether `p_hat - epsilon > 0.5`, where:
   - `p_hat` is the observed mode frequency (e.g., 0.8 after 5 samples)
   - `epsilon = sqrt(log(2/delta) / (2*k))` is the Hoeffding confidence half-width
   - `delta` is a user-configurable confidence parameter (default: 0.05)
4. If the bound confirms the mode is dominant, stop sampling for this question.

The Hoeffding bound guarantees that stopping early does not compromise the statistical validity of the results. The bound is conservative -- it only stops when there is strong evidence that additional samples would not change the mode.

**Cost savings:** In practice, "easy" questions (where the model is highly consistent) stop after 3-5 samples, while "hard" questions (where answers vary) use all K. Across a typical benchmark, this saves 40-60% of API calls compared to fixed-K sampling, without affecting the reliability guarantee.

The `delta` parameter controls how aggressive the stopping is. Smaller delta (e.g., 0.01) means more conservative stopping (fewer early stops, higher certainty). Larger delta (e.g., 0.10) means more aggressive stopping (more early stops, slightly lower certainty). The default of 0.05 provides a good balance.
