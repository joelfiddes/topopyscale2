"""Shipped code opens text files with an explicit encoding (Windows).

On Windows the default text encoding is the locale's (often cp1252), not UTF-8, so a
config with a point named "Zürich" would load garbled. Every builtin open() in text mode
and every Path.read_text()/write_text() in a shipped module must pass encoding=.
"""

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("allowlist", REPO / "scripts/release/allowlist.py")


def _shipped_modules():
    if _spec is None or not (REPO / "scripts/release/allowlist.py").exists():
        # A snapshot ships only shipped modules: check them all.
        return sorted((REPO / "topopyscale2").rglob("*.py"))
    allowlist = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(allowlist)
    rules = allowlist.load()
    return sorted(p for p in (REPO / "topopyscale2").rglob("*.py")
                  if allowlist.allowed(p.relative_to(REPO).as_posix(), rules))


def _violations(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {k.arg for k in node.keywords}
        if "encoding" in kwargs:
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if not (isinstance(mode, str) and "b" in mode):
                yield node.lineno, "open()"
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            yield node.lineno, f".{func.attr}()"


def test_shipped_text_io_names_its_encoding():
    modules = _shipped_modules()
    assert len(modules) > 40
    bad = [f"{p.relative_to(REPO)}:{line} {what}" for p in modules for line, what in _violations(p)]
    assert not bad, "text I/O without encoding= (breaks non-ASCII on Windows):\n  " + "\n  ".join(bad)


def test_detector_flags_a_bare_open(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("open('x')\nopen('y', 'rb')\nopen('z', encoding='utf-8')\nP.read_text()\n", encoding="utf-8")
    assert [w for _, w in _violations(probe)] == ["open()", ".read_text()"]
