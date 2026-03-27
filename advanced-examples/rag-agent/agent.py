"""Country facts agent with two tools: retriever + calculator.

A FastAPI agent that:
1. Searches a local knowledge base of country facts
2. Can perform math calculations
3. Uses an LLM to generate natural-language answers from retrieved context

The LLM generation step introduces natural variance in phrasing, which is
what TrustGate's self-consistency sampling measures.

Run: uvicorn agent:app --port 8000
Test: curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
      -d '{"query": "What is the population density of Japan?"}'
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI

app = FastAPI(title="Country Facts Agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LLM_URL = os.environ.get(
    "AGENT_LLM_URL", "https://api.openai.com/v1/chat/completions"
)
_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "gpt-4.1-nano")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
_LLM_TEMPERATURE = float(os.environ.get("AGENT_LLM_TEMPERATURE", "0.7"))

# ---------------------------------------------------------------------------
# Knowledge base (loaded once at startup)
# ---------------------------------------------------------------------------

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_DOCS: dict[str, str] = {}

for f in _KNOWLEDGE_DIR.glob("*.md"):
    _DOCS[f.stem.replace("_", " ")] = f.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool 1: Retriever — keyword search over country fact files
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Simple keyword retriever. Scores docs by word overlap with query."""
    query_words = set(query.lower().split())
    scored = []
    for name, content in _DOCS.items():
        doc_words = set(content.lower().split())
        overlap = len(query_words & doc_words)
        # Boost if country name appears in query
        if name.lower() in query.lower():
            overlap += 10
        scored.append((overlap, name, content))

    scored.sort(reverse=True)
    return [
        {"country": name, "content": content}
        for _, name, content in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Tool 2: Calculator — evaluates math expressions
# ---------------------------------------------------------------------------

_SAFE_NAMES = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    # Clean the expression
    expr = expression.strip()
    # Allow only digits, operators, parentheses, dots, spaces
    if not re.match(r'^[\d\s\+\-\*/\.\(\)%eE]+$', expr):
        return f"Error: invalid expression '{expr}'"
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.2f}"
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Agent logic — decide which tools to use, combine results
# ---------------------------------------------------------------------------

_CALC_KEYWORDS = {
    "density", "per capita", "percentage", "percent", "ratio",
    "divide", "multiply", "calculate", "how much", "how many times",
    "average", "total", "sum", "difference", "compared to",
}


def _build_context(query: str) -> str:
    """Retrieve docs and optionally run calculations to build answer context."""
    query_lower = query.lower()

    # Always retrieve relevant docs
    docs = retrieve(query)
    context_parts = [
        f"From {d['country']}:\n{d['content']}" for d in docs
    ]

    # Check if calculation is needed
    needs_calc = any(kw in query_lower for kw in _CALC_KEYWORDS)

    if needs_calc:
        # Extract numbers from context for calculation
        full_context = "\n\n".join(context_parts)
        numbers = _extract_numbers(full_context, query_lower)

        if "density" in query_lower and numbers.get("population") and numbers.get("area"):
            pop = numbers["population"]
            area = numbers["area"]
            result = calculate(f"{pop} / {area}")
            context_parts.append(
                f"\nCalculation: population {pop:,.0f} / area {area:,.0f} km² = {result} people per km²"
            )

        elif "per capita" in query_lower and numbers.get("gdp") and numbers.get("population"):
            gdp = numbers["gdp"]
            pop = numbers["population"]
            result = calculate(f"{gdp} / {pop}")
            context_parts.append(
                f"\nCalculation: GDP ${gdp:,.0f} / population {pop:,.0f} = ${result} per capita"
            )

        elif ("percentage" in query_lower or "percent" in query_lower or "ratio" in query_lower):
            if len(numbers) >= 2:
                vals = list(numbers.values())
                result = calculate(f"{vals[0]} / {vals[1]} * 100")
                context_parts.append(
                    f"\nCalculation: {vals[0]:,.0f} / {vals[1]:,.0f} = {result}%"
                )

    return "\n\n".join(context_parts)


async def _llm_answer(query: str, context: str) -> str:
    """Use an LLM to generate a concise answer from retrieved context."""
    if not _LLM_API_KEY:
        # Fallback: return raw context if no LLM configured
        return context

    system_prompt = (
        "You are a helpful assistant that answers questions based on provided context. "
        "Give a concise, direct answer. If the context doesn't contain enough information, "
        "say so. Do not make up facts not in the context."
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _LLM_URL,
            headers={
                "Authorization": f"Bearer {_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _LLM_MODEL,
                "temperature": _LLM_TEMPERATURE,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _extract_numbers(context: str, query: str) -> dict[str, float]:
    """Extract key numbers from context based on query keywords."""
    numbers: dict[str, float] = {}

    # Population
    pop_match = re.search(r"Population:\s*([\d,.]+)\s*million", context)
    if pop_match:
        numbers["population"] = float(pop_match.group(1).replace(",", "")) * 1_000_000

    # Area
    area_match = re.search(r"Area:\s*([\d,.]+)\s*km", context)
    if area_match:
        numbers["area"] = float(area_match.group(1).replace(",", ""))

    # GDP
    gdp_match = re.search(r"GDP:\s*\$([\d,.]+)\s*trillion", context)
    if gdp_match:
        numbers["gdp"] = float(gdp_match.group(1).replace(",", "")) * 1_000_000_000_000

    # Life expectancy
    life_match = re.search(r"Life expectancy:\s*([\d.]+)", context)
    if life_match:
        numbers["life_expectancy"] = float(life_match.group(1))

    return numbers


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask(request: dict) -> dict:
    """Answer a question using the country facts agent."""
    query = request.get("query", "")
    if not query:
        return {"answer": "Please provide a query."}

    context = _build_context(query)
    answer = await _llm_answer(query, context)
    return {"answer": answer}


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "ok",
        "docs_loaded": len(_DOCS),
        "llm_configured": bool(_LLM_API_KEY),
        "llm_model": _LLM_MODEL,
    }
