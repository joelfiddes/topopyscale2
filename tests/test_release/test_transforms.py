"""The export's public variants exist, are allowlisted, and the packaging transforms apply."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if not (REPO / "scripts" / "release" / "transforms.py").exists():
    # A public snapshot ships without the export tooling: nothing is left to move.
    pytest.skip("export tooling not in this tree (a public snapshot)", allow_module_level=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "release" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


transforms = _load("transforms")
allowlist = _load("allowlist")
# In an exported snapshot the public variants have already been moved into place.
SNAPSHOT = not (REPO / "README.public.md").exists()


@pytest.mark.parametrize("src", sorted(transforms.MOVES))
def test_every_public_variant_exists_and_ships(src):
    if SNAPSHOT:
        pytest.skip("already moved: this is an exported snapshot")
    assert (REPO / src).is_file(), f"{src} is missing"
    assert allowlist.allowed(src, allowlist.load()), f"{src} is not on the export allowlist"


@pytest.mark.parametrize("rel", sorted(transforms.TRANSFORMS))
def test_every_transform_applies_to_the_current_file(rel):
    if SNAPSHOT:
        pytest.skip("already transformed: this is an exported snapshot")
    transforms.TRANSFORMS[rel]((REPO / rel).read_text(encoding="utf-8"))   # raises if an anchor moved
