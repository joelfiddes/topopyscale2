"""Tests for CLI."""

from pathlib import Path

from typer.testing import CliRunner

import topopyscale2
from tests._release import requires
from topopyscale2.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TopoPyScale" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert topopyscale2.__version__ in result.output

    def test_info_command(self):
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "central_asia_basic.yaml"
        )
        result = runner.invoke(app, ["info", "--config", yaml_path])
        assert result.exit_code == 0
        assert "68.0" in result.output  # bbox west
        assert "google" in result.output  # backend

    def test_setup_help(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    @requires("topopyscale2.cli.stations_cli")
    def test_validate_help(self):
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
