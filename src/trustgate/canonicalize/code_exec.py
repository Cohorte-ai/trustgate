"""Sandboxed code execution (pass/fail)."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from trustgate.canonicalize import Canonicalizer, register_canonicalizer

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Extract the first Python code block, or treat the whole text as code."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # If no code fence, assume the entire text is code
    return text.strip()


@register_canonicalizer("code_exec")
class CodeExecCanonicalizer(Canonicalizer):
    """Execute code in a sandboxed subprocess and return "pass" or "fail"."""

    def __init__(self, timeout: float = 10.0, **kwargs: object) -> None:
        self.timeout = timeout

    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        if not text:
            return "fail"

        code = _extract_code(text)
        if not code:
            return "fail"

        return self._run_sandboxed(code)

    def _run_sandboxed(self, code: str) -> str:
        """Run Python code in a subprocess with timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "solution.py"
            script.write_text(code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env={"PATH": "", "HOME": tmpdir},
                )
                return "pass" if result.returncode == 0 else "fail"

            except subprocess.TimeoutExpired:
                return "fail"
            except Exception:
                return "fail"
