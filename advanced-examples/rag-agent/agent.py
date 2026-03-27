"""Country facts agent with two tools: retriever + calculator.

A simple FastAPI agent that:
1. Searches a local knowledge base of country facts
2. Can perform math calculations
3. Combines both to answer questions

Run: uvicorn agent:app --port 8000
Test: curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
      -d '{"query": "What is the population density of Japan?"}'
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Country Facts Agent")

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


def agent_answer(query: str) -> str:
    """Process a query using retriever + calculator as needed."""
    query_lower = query.lower()

    # Always retrieve relevant docs
    docs = retrieve(query)
    context = "\n\n".join(
        f"## {d['country']}\n{d['content']}" for d in docs
    )

    # Check if calculation is needed
    needs_calc = any(kw in query_lower for kw in _CALC_KEYWORDS)

    if needs_calc:
        # Extract numbers from context for calculation
        numbers = _extract_numbers(context, query_lower)
        calc_result = None

        if "density" in query_lower or "per capita" in query_lower:
            # population / area or gdp / population
            if "density" in query_lower and len(numbers) >= 2:
                pop = numbers.get("population")
                area = numbers.get("area")
                if pop and area:
                    calc_result = calculate(f"{pop} / {area}")
                    return f"Based on the data: population {pop:,.0f} / area {area:,.0f} km² = {calc_result} people per km²"

            if "per capita" in query_lower and len(numbers) >= 2:
                gdp = numbers.get("gdp")
                pop = numbers.get("population")
                if gdp and pop:
                    calc_result = calculate(f"{gdp} / {pop}")
                    return f"Based on the data: GDP ${gdp:,.0f} / population {pop:,.0f} = ${calc_result} per capita"

        if "percentage" in query_lower or "percent" in query_lower or "ratio" in query_lower:
            if len(numbers) >= 2:
                vals = list(numbers.values())
                calc_result = calculate(f"{vals[0]} / {vals[1]} * 100")
                return f"Ratio: {vals[0]:,.0f} / {vals[1]:,.0f} = {calc_result}%"

        if "compared to" in query_lower and len(numbers) >= 2:
            vals = list(numbers.values())
            calc_result = calculate(f"{vals[0]} / {vals[1]}")
            return f"Comparison: {vals[0]:,.0f} / {vals[1]:,.0f} = {calc_result}x"

    # Pure retrieval — answer from context
    # Return the most relevant doc's key facts
    if docs:
        top_doc = docs[0]
        return f"From {top_doc['country']} data:\n{top_doc['content']}"

    return "I don't have information about that."


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
    answer = agent_answer(query)
    return {"answer": answer}


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "ok", "docs_loaded": len(_DOCS)}
