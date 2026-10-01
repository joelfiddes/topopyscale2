"""Configs load when optional features are not installed (#214).

A public v1 snapshot ships only the downscaling core. A config carrying a block for
a feature that is absent (e.g. ``validation_lab:``) must still load: the block is
ignored and named once in a warning, instead of failing every run.
"""

import logging
import sys

import pytest

from tests._release import requires
from topopyscale2.config import schema
from topopyscale2.config.schema import TPS2Config, feature_installed

DOMAIN = {"bbox": [9.7, 46.7, 9.9, 46.9]}
VALIDATION_MODULES = ("topopyscale2.validation", "topopyscale2.validation.registry",
                      "topopyscale2.validation.protocols")


@pytest.fixture
def validation_absent(monkeypatch):
    """Make `import topopyscale2.validation...` fail, as in a v1 snapshot."""
    for m in VALIDATION_MODULES:
        monkeypatch.setitem(sys.modules, m, None)
    monkeypatch.setattr(schema, "feature_installed",
                        lambda mod: not mod.startswith("topopyscale2.validation"))


def test_default_config_loads_without_validation(validation_absent):
    # The default ValidationLabConfig runs a registry validator on every load.
    assert TPS2Config(domain=DOMAIN).domain.bbox == DOMAIN["bbox"]


def test_absent_feature_block_is_ignored_with_one_warning(validation_absent, caplog):
    with caplog.at_level(logging.WARNING, logger="topopyscale2.config.schema"):
        cfg = TPS2Config(domain=DOMAIN,
                         validation_lab={"datasets": ["not_a_real_dataset"],
                                         "protocols": ["not_a_real_protocol"]})
    assert cfg.validation_lab.datasets == ["not_a_real_dataset"]
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and "'validation_lab:'" in msgs[0], msgs


@requires("topopyscale2.validation")
def test_no_warning_when_features_are_installed(caplog):
    with caplog.at_level(logging.WARNING, logger="topopyscale2.config.schema"):
        TPS2Config(domain=DOMAIN, validation={})
    assert not caplog.records


@requires("topopyscale2.validation")
def test_installed_feature_is_still_validated():
    with pytest.raises(ValueError, match="unknown dataset id"):
        TPS2Config(domain=DOMAIN, validation_lab={"datasets": ["not_a_real_dataset"]})


def test_feature_installed():
    assert feature_installed("topopyscale2.core")
    assert not feature_installed("topopyscale2.no_such_package.sub")


def test_every_feature_block_is_a_config_field():
    assert set(schema.FEATURE_MODULES) <= set(TPS2Config.model_fields)
