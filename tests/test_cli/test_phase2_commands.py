"""Tests for Phase 2 CLI commands.

Tests the new CLI commands added in Phase 2:
- qc stations command
- qc report command
- app run command
- app validate command
- Updated validate command
- Updated info command with Phase 2 sections
"""

import tempfile
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from tests._release import requires
from topopyscale2.cli.main import app

runner = CliRunner()


@requires("topopyscale2.cli.stations_cli")
class TestQCCommandHelp:
    """Tests for QC command help text."""

    def test_qc_help(self):
        """QC command group shows help."""
        result = runner.invoke(app, ["qc", "--help"])
        assert result.exit_code == 0
        assert "Quality control" in result.output

    def test_qc_stations_help(self):
        """QC stations command shows help."""
        result = runner.invoke(app, ["qc", "stations", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--variable" in result.output
        assert "temperature" in result.output

    def test_qc_report_help(self):
        """QC report command shows help."""
        result = runner.invoke(app, ["qc", "report", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output


@requires("topopyscale2.cli.snow_cli")
class TestAppCommandHelp:
    """Tests for app command help text."""

    def test_app_help(self):
        """App command group shows help."""
        result = runner.invoke(app, ["app", "--help"])
        assert result.exit_code == 0
        assert "Application layer" in result.output

    def test_app_run_help(self):
        """App run command shows help."""
        result = runner.invoke(app, ["app", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output
        assert "fsm" in result.output.lower()

    def test_app_validate_help(self):
        """App validate command shows help."""
        result = runner.invoke(app, ["app", "validate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--obs" in result.output


@requires("topopyscale2.cli.stations_cli")
class TestValidateCommandHelp:
    """Tests for updated validate command."""

    def test_validate_help(self):
        """Validate command shows Phase 2 options."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--stations" in result.output
        assert "--variable" in result.output


class TestInfoCommandPhase2:
    """Tests for info command with Phase 2 configuration."""

    def test_info_polygon_mode(self):
        """Info command shows polygon configuration."""
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        result = runner.invoke(app, ["info", "--config", yaml_path])
        assert result.exit_code == 0
        assert "polygons" in result.output.lower()
        assert "Elevation bands" in result.output
        assert "Aspect classes" in result.output

    def test_info_wind_config(self):
        """Info command shows wind configuration."""
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        result = runner.invoke(app, ["info", "--config", yaml_path])
        assert result.exit_code == 0
        assert "winstral" in result.output.lower()

    def test_info_redistribution(self):
        """Info command shows redistribution configuration."""
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        result = runner.invoke(app, ["info", "--config", yaml_path])
        assert result.exit_code == 0
        assert "Redistribution" in result.output

    def test_info_storage(self):
        """Info command shows storage configuration."""
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        result = runner.invoke(app, ["info", "--config", yaml_path])
        assert result.exit_code == 0
        assert "Storage" in result.output
        assert "zarr" in result.output.lower()


class TestRunCommandPhase2:
    """Tests for run command Phase 2 features."""

    def test_run_dry_run(self):
        """Run command with --dry-run shows config summary."""
        yaml_path = str(
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        result = runner.invoke(app, ["run", "--config", yaml_path, "--dry-run"])
        assert result.exit_code == 0
        assert "Configuration:" in result.output
        assert "Dry run mode" in result.output

    def test_run_help_shows_phase2_features(self):
        """Run command help mentions Phase 2 features."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--backend" in result.output


@requires("topopyscale2.cli.stations_cli")
class TestQCStationsCommand:
    """Tests for QC stations command execution."""

    def test_qc_stations_missing_input(self):
        """QC stations errors on missing input file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name
        result = runner.invoke(app, [
            "qc", "stations",
            "--input", "/nonexistent/file.csv",
            "--output", output_path,
            "--variable", "temperature"
        ])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_qc_stations_basic_execution(self):
        """QC stations runs with valid input."""
        # Create test input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("2020-01-01 00:00:00,10.0\n")
            f.write("2020-01-01 01:00:00,11.0\n")
            f.write("2020-01-01 02:00:00,12.0\n")
            input_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        result = runner.invoke(app, [
            "qc", "stations",
            "--input", input_path,
            "--output", output_path,
            "--variable", "temperature"
        ])
        assert result.exit_code == 0
        assert "QC Summary" in result.output
        assert "Good" in result.output

        # Verify output was written
        output_df = pd.read_csv(output_path)
        assert "qc_flag" in output_df.columns

        # Cleanup
        Path(input_path).unlink()
        Path(output_path).unlink()

    def test_qc_stations_unknown_variable(self):
        """QC stations errors on unknown variable type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("2020-01-01 00:00:00,10.0\n")
            input_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        result = runner.invoke(app, [
            "qc", "stations",
            "--input", input_path,
            "--output", output_path,
            "--variable", "unknown_variable"
        ])
        assert result.exit_code == 1
        assert "Unknown variable" in result.output

        # Cleanup
        Path(input_path).unlink()


@requires("topopyscale2.cli.stations_cli")
class TestQCReportCommand:
    """Tests for QC report command execution."""

    def test_qc_report_missing_input(self):
        """QC report errors on missing input file."""
        result = runner.invoke(app, [
            "qc", "report",
            "--input", "/nonexistent/file.csv"
        ])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_qc_report_missing_qc_columns(self):
        """QC report errors if input lacks QC columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("2020-01-01 00:00:00,10.0\n")
            input_path = f.name

        result = runner.invoke(app, [
            "qc", "report",
            "--input", input_path
        ])
        assert result.exit_code == 1
        assert "qc_flag" in result.output

        Path(input_path).unlink()

    def test_qc_report_basic_execution(self):
        """QC report runs with valid QC output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value,qc_flag,qc_reason\n")
            f.write("2020-01-01 00:00:00,10.0,0,\n")
            f.write("2020-01-01 01:00:00,11.0,1,temporal\n")
            f.write("2020-01-01 02:00:00,100.0,2,range\n")
            input_path = f.name

        result = runner.invoke(app, [
            "qc", "report",
            "--input", input_path
        ])
        assert result.exit_code == 0
        assert "Overall Summary" in result.output
        assert "Good:" in result.output
        assert "Suspect:" in result.output
        assert "Rejected:" in result.output

        Path(input_path).unlink()
