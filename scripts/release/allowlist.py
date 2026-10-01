"""Read ``export_allowlist.txt``: which repository paths ship in a public snapshot.

One source of truth for the export (#210) and the v1 import-boundary test (#211).
Rules are applied in order and the last match wins, so ``!pattern`` can carve an
exclusion out of an earlier directory include. A path no rule matches is private.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

ALLOWLIST = Path(__file__).resolve().parent / "export_allowlist.txt"


def load(path: Path = ALLOWLIST) -> list[tuple[bool, str]]:
    """Return ``(include, pattern)`` rules; raise if the file is missing or has no includes."""
    rules = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rules.append((False, line[1:].strip()) if line.startswith("!") else (True, line))
    # Fail closed: an allowlist that includes nothing would publish (and check) nothing.
    if not any(inc for inc, _ in rules):
        raise ValueError(f"allowlist {path} includes no paths")
    return rules


def _matches(rel: str, pat: str) -> bool:
    if pat.endswith("/"):
        return rel.startswith(pat)
    return rel == pat or fnmatch.fnmatchcase(rel, pat)


def allowed(rel: str, rules: list[tuple[bool, str]]) -> bool:
    """True if the posix repo-relative path ``rel`` ships."""
    verdict = False
    for include, pat in rules:
        if _matches(rel, pat):
            verdict = include
    return verdict
