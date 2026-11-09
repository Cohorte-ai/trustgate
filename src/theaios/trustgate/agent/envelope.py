"""Validity envelope: a fingerprint of everything that can change a qualification result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from theaios.trustgate import __version__


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def grader_digest() -> str:
    """Hash of the code that grades runs, so a grader change expires results even when the
    version number does not move."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def envelope_components(
    base_dir: Path,
    *,
    model: str | None,
    file_globs: list[str],
    extra: dict[str, Any],
    scenario_files: list[Path],
    target: dict[str, Any],
) -> dict[str, Any]:
    """What the result depends on. Target headers are excluded (they may hold secrets)."""
    files: dict[str, str] = {}
    for pattern in file_globs:
        for match in sorted(base_dir.glob(pattern)):
            # Before Python 3.13, "dir/**" matches directories only: take their contents.
            members = sorted(match.rglob("*")) if match.is_dir() else [match]
            for p in members:
                if p.is_file():
                    files[p.relative_to(base_dir).as_posix()] = _sha(p.read_bytes())
    target_public = {k: v for k, v in target.items() if k != "headers"}
    return {
        "trustgate": __version__,
        "grader": grader_digest(),
        "model": model,
        "files": files,
        "extra": extra,
        "scenarios": {p.name: _sha(p.read_bytes()) for p in sorted(scenario_files)},
        "target": target_public,
    }


def fingerprint(components: dict[str, Any]) -> str:
    blob = json.dumps(components, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:24]


def changed_components(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Human-readable list of what differs between two envelopes."""
    changes: list[str] = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict) and key in ("files", "scenarios"):
            for name in sorted(set(a) | set(b)):
                if name not in a:
                    changes.append(f"{key}: added {name}")
                elif name not in b:
                    changes.append(f"{key}: removed {name}")
                elif a[name] != b[name]:
                    changes.append(f"{key}: changed {name}")
        else:
            changes.append(f"{key}: {a!r} -> {b!r}")
    return changes
