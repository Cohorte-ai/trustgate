# Example: Certifying a RAG Agent with TrustGate

This example shows how to certify a real AI agent — not just a raw LLM, but a system with **three components**:

1. **Retriever** — searches a local knowledge base of country facts (deterministic)
2. **Calculator** — evaluates math expressions for derived metrics (deterministic)
3. **LLM generator** — produces natural-language answers from retrieved context (stochastic)

The retriever and calculator are deterministic, but the LLM generator introduces variance in phrasing — exactly what TrustGate's self-consistency sampling measures.

## Setup

### macOS / Linux

```bash
cd advanced-examples/rag-agent
python3 -m venv .venv
source .venv/bin/activate
pip install theaios-trustgate fastapi uvicorn
export LLM_API_KEY="sk-your-key-here"
```

### Windows (PowerShell)

```powershell
cd advanced-examples\rag-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install theaios-trustgate fastapi uvicorn
$env:LLM_API_KEY="sk-your-key-here"
```

## Run the agent

Start the agent in one terminal:

### macOS / Linux

```bash
cd advanced-examples/rag-agent
source .venv/bin/activate
export LLM_API_KEY="sk-your-key-here"
uvicorn agent:app --port 8000
```

### Windows (PowerShell)

```powershell
cd advanced-examples\rag-agent
.venv\Scripts\Activate.ps1
$env:LLM_API_KEY="sk-your-key-here"
uvicorn agent:app --port 8000
```

The agent uses `LLM_API_KEY` both for its own answer generation and for TrustGate's canonicalization.

Verify it works:

### macOS / Linux

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Windows (PowerShell)

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"
```

Expected: a natural-language answer like `{"answer": "The capital of France is Paris."}`

## Certify

Open a **second terminal**, activate the same venv, and set the API key:

### macOS / Linux

```bash
cd advanced-examples/rag-agent
source .venv/bin/activate
export LLM_API_KEY="sk-your-key-here"
trustgate certify
```

### Windows (PowerShell)

```powershell
cd advanced-examples\rag-agent
.venv\Scripts\Activate.ps1
$env:LLM_API_KEY="sk-your-key-here"
trustgate certify
```

You'll see:
1. API latency measurement
2. Pre-flight cost estimate (small cost for both agent LLM calls and canonicalization)
3. Certification result with reliability level, M*, and coverage

## Try the hard dataset

The default `questions.csv` has easy factual lookups (high reliability). Switch to the hard dataset to see realistic variation:

Edit `trustgate.yaml`:

```yaml
questions:
  # file: "questions.csv"           # easy: factual lookups
  file: "questions_hard.csv"        # hard: multi-hop, ambiguous, out-of-scope
```

Then re-run:

```bash
trustgate cache clear
trustgate certify
```

The hard dataset includes multi-country aggregation, multi-step math, cross-field filtering, and edge cases — questions that will produce lower reliability scores.

## What to observe

- **Easy questions**: high self-consistency — the LLM generates the same factual answer in different phrasings
- **Hard questions**: lower self-consistency — the LLM may compute different values or miss steps
- **Calibration UI**: run `trustgate calibrate --serve` to see the raw model responses — you'll notice the same answer phrased differently each time
- **Capability gap**: fraction of questions the agent can't answer (e.g., countries not in the knowledge base)

## Configuring the agent's LLM

The agent uses environment variables for its LLM configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | API key for the LLM |
| `AGENT_LLM_URL` | `https://api.openai.com/v1/chat/completions` | LLM endpoint URL |
| `AGENT_LLM_MODEL` | `gpt-4.1-nano` | Model to use for answer generation |
| `AGENT_LLM_TEMPERATURE` | `0.7` | Temperature for generation (higher = more variance) |

## Customizing

- **Add more countries**: drop a markdown file in `knowledge/`
- **Add more questions**: edit `questions.csv` or `questions_hard.csv`
- **Change the LLM judge**: edit `trustgate.yaml` → `judge_endpoint`
- **Change K**: edit `trustgate.yaml` → `sampling.k_fixed`, or type a number at the prompt
- **Increase variance**: set `AGENT_LLM_TEMPERATURE=1.0`
