"""v1 import boundary (#211): modules that ship must not import modules that don't.

The shipped set is the package section of ``scripts/release/export_allowlist.txt``,
the same file the public export uses. Imports are read statically (AST), so the
check needs none of the optional dependencies and sees imports inside functions
too: a lazy import of a missing module still fails when that code path runs.

Exempt: imports under ``if TYPE_CHECKING:`` and imports inside a ``try`` whose
handler catches ImportError/ModuleNotFoundError (optional features by design).

Edges that exist today are listed in ``import_boundary_known.txt``. The test fails on
any NEW edge, and on any listed edge that no longer exists, so the list only shrinks.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = "topopyscale2"
# Shipped test files are checked too: a shipped test that imports unshipped code
# (a module, or another test's helpers) would fail in the snapshot.
ROOTS = (PKG, "tests")
KNOWN = Path(__file__).with_name("import_boundary_known.txt")
# Compiled extension modules: no .py source, shipped when the package is.
COMPILED = {"topopyscale2.core._rust_kernels"}

_spec = importlib.util.spec_from_file_location("allowlist", REPO / "scripts/release/allowlist.py")
allowlist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(allowlist)


def _module_file(name: str) -> Path | None:
    base = REPO / Path(*name.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _module_name(path: Path) -> str:
    parts = path.relative_to(REPO).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _catches_import_error(node: ast.Try) -> bool:
    names = set()
    for h in node.handlers:
        if h.type is None:
            return True
        for t in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]):
            if isinstance(t, ast.Name):
                names.add(t.id)
    return bool(names & {"ImportError", "ModuleNotFoundError", "Exception"})


def _imports(path: Path) -> set[str]:
    """Package-internal modules ``path`` imports, excluding exempt imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    here = _module_name(path)
    is_pkg = path.name == "__init__.py"
    found: set[str] = set()

    def resolve_from(node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = here.split(".") if is_pkg else here.split(".")[:-1]
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for n in node.orelse:
                visit(n)
            return
        if isinstance(node, ast.Try) and _catches_import_error(node):
            for n in node.handlers + node.orelse + node.finalbody:
                visit(n)
            return
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = resolve_from(node)
            for a in node.names:
                sub = f"{mod}.{a.name}"
                # `from pkg import sub` imports the submodule when one exists.
                found.add(sub if (_module_file(sub) or sub in COMPILED) else mod)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return {m for m in found if any(m == r or m.startswith(r + ".") for r in ROOTS)}


def _ships(name: str, rules) -> bool:
    if name in COMPILED:
        return _ships(name.rsplit(".", 1)[0], rules)
    f = _module_file(name)
    return f is not None and allowlist.allowed(f.relative_to(REPO).as_posix(), rules)


def _v1_modules(rules) -> list[Path]:
    return sorted(p for root in ROOTS for p in (REPO / root).rglob("*.py")
                  if allowlist.allowed(p.relative_to(REPO).as_posix(), rules))


def current_edges() -> set[str]:
    rules = allowlist.load()
    edges = set()
    for path in _v1_modules(rules):
        src = _module_name(path)
        for tgt in _imports(path):
            if not _ships(tgt, rules):
                edges.add(f"{src} -> {tgt}")
        # A shipped module needs every parent package's __init__ to ship too.
        parts = src.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if not _ships(parent, rules):
                edges.add(f"{src} -> {parent} (parent package)")
    return edges


def known_edges() -> set[str]:
    lines = KNOWN.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def test_allowlist_defines_a_real_v1_package():
    """Gate-absent case: a missing/empty allowlist must not pass as 'no violations'."""
    mods = _v1_modules(allowlist.load())
    assert len(mods) >= 40, f"only {len(mods)} v1 modules; allowlist looks truncated"
    for core in ("core/downscale.py", "domain.py", "cli/main.py"):
        assert (REPO / PKG / core) in mods, f"{core} missing from the v1 allowlist"


def test_empty_allowlist_refuses(tmp_path):
    empty = tmp_path / "allow.txt"
    empty.write_text("# nothing\n!topopyscale2/\n")
    with pytest.raises(ValueError):
        allowlist.load(empty)


def test_no_new_boundary_edges():
    new = sorted(current_edges() - known_edges())
    assert not new, ("v1 modules import modules that do not ship. Make the import lazy "
                     "and optional, move the code, or (deliberately) extend the allowlist:\n  "
                     + "\n  ".join(new))


def test_known_edges_list_only_shrinks():
    stale = sorted(known_edges() - current_edges())
    assert not stale, ("these edges are fixed; delete them from "
                       f"{KNOWN.name}:\n  " + "\n  ".join(stale))


def test_detector_sees_a_violation(tmp_path, monkeypatch):
    """The analyser itself must flag a plain, a relative and a function-level import."""
    f = REPO / PKG / "_boundary_probe_tmp.py"
    f.write_text("import topopyscale2.da.enkf\nfrom ..ml import losses\n"
                 "def f():\n    from topopyscale2.dashboard import registry\n"
                 "if TYPE_CHECKING:\n    import topopyscale2.validation\n")
    try:
        got = _imports(f)
    finally:
        f.unlink()
    # A missing submodule resolves to its package, so match by prefix: the probe must
    # work in a snapshot where these packages do not exist.
    assert any(m.startswith("topopyscale2.da") for m in got), got
    assert any(m.startswith("topopyscale2.dashboard") for m in got), got
    assert "topopyscale2.validation" not in got
