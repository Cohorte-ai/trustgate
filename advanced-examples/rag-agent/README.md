# Example: Certifying a RAG Agent with TrustGate

This example shows how to certify a real AI agent — not just a raw LLM, but a system with **two tools**:

1. **Retriever** — searches a local knowledge base of country facts
2. **Calculator** — evaluates math expressions for derived metrics

The agent handles three types of questions:
- **Pure retrieval**: "What is the capital of France?" → searches docs
- **Pure calculation**: "What is 15% of 340 million?" → uses calculator
- **Hybrid**: "What is the population density of Japan?" → retrieves population + area, then divides

This is a realistic pattern — most production agents combine knowledge retrieval with computation.

## Setup

### macOS / Linux

```bash
pip install theaios-trustgate fastapi uvicorn
export LLM_API_KEY="sk-your-key-here"
```

### Windows (PowerShell)

```powershell
pip install theaios-trustgate fastapi uvicorn
$env:LLM_API_KEY="sk-your-key-here"
```

## Run the agent

Start the agent in one terminal:

### macOS / Linux

```bash
cd advanced-examples/rag-agent
uvicorn agent:app --port 8000
```

### Windows (PowerShell)

```powershell
cd advanced-examples\rag-agent
uvicorn agent:app --port 8000
```

Verify it works:

### macOS / Linux

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Windows (PowerShell)

```powershell
curl.exe -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"
```

Expected: `{"answer": "From france data:\n# France\n..."}`

## Certify

In another terminal:

### macOS / Linux

```bash
cd advanced-examples/rag-agent
trustgate certify
```

### Windows (PowerShell)

```powershell
cd advanced-examples\rag-agent
trustgate certify
```

You'll see:
1. API latency measurement (~instant since it's local)
2. Pre-flight cost estimate ($0 for the agent, small cost for LLM canonicalization)
3. Certification result

## What to observe

- **Retrieval questions** (q01-q12): high consistency — the agent always finds the same doc
- **Calculation questions** (q13-q20): high consistency — same inputs → same calculation
- **Comparison questions** (q21-q25): may vary — the retriever might rank docs differently, leading to different numbers being extracted
- **Capability gap**: questions the agent can't answer (if any) — the correct answer never appears in K samples

## Certify each component

You can also certify the retriever and calculator independently:

```bash
# Certify just the retriever (which docs are returned)
# → Would need a custom canonicalizer that extracts document names

# Certify just the calculator (which expression is generated)
# → Would need a custom canonicalizer that extracts the math expression
```

This is the "certify each component" pattern from the TrustGate docs.

## Customizing

- **Add more countries**: drop a markdown file in `knowledge/`
- **Add more questions**: edit `questions.csv`
- **Change the LLM judge**: edit `trustgate.yaml` → `judge_endpoint`
- **Change K**: edit `trustgate.yaml` → `sampling.k_fixed`, or use `--alpha` at the CLI
