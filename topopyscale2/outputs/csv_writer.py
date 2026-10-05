"""Generic CSV tabular format output writer."""

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from topopyscale2.outputs.base import CF_ATTRIBUTES
from topopyscale2.spatial.units import SpatialUnit


class CSVWriter:
    """Write forcing as CSV tabular format.

    CSV output is useful for:
    - Easy inspection and visualization in spreadsheets
    - Import into custom models and scripts
    - Data exchange with non-Python tools
    - Debugging and validation

    Supports multiple output modes:
    - One file per unit (long or wide format)
    - Single combined file (stacked or multi-column)
    """

    model_name = "generic"
    file_format = "csv"

    def __init__(
        self,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        float_format: str = "%.6g",
        include_metadata_header: bool = True,
        single_file: bool = False,
        wide_format: bool = False,
        variables: Optional[list[str]] = None,
        na_rep: str = "NaN",
    ):
        """Initialize CSV writer.

        Parameters
        ----------
        date_format : str
            Date format string for time column.
        float_format : str
            Float format string for numeric values.
        include_metadata_header : bool
            Whether to include metadata comment lines at top of file.
        single_file : bool
            If True, write all units to a single file.
        wide_format : bool
            If True and single_file=True, use wide format with one column
            per unit per variable. Otherwise use long format with unit_id column.
        variables : list[str], optional
            Subset of variables to include. If None, include all.
        na_rep : str
            String representation for missing values.
        """
        self.date_format = date_format
        self.float_format = float_format
        self.include_metadata_header = include_metadata_header
        self.single_file = single_file
        self.wide_format = wide_format
        self.variables = variables
        self.na_rep = na_rep

    def _get_variable_info(self, var_name: str) -> dict:
        """Get variable metadata from CF attributes.

        Parameters
        ----------
        var_name : str
            Variable name.

        Returns
        -------
        dict
            Variable attributes.
        """
        if var_name in CF_ATTRIBUTES:
            return CF_ATTRIBUTES[var_name]
        return {"long_name": var_name, "units": "unknown"}

    def _write_metadata_header(
        self,
        f,
        unit: Optional[SpatialUnit] = None,
        variables: list[str] = None,
    ) -> None:
        """Write metadata header comments.

        Parameters
        ----------
        f : file handle
            Open file to write to.
        unit : SpatialUnit, optional
            Spatial unit for metadata.
        variables : list[str]
            Variables included in output.
        """
        f.write("# TopoPyScale 2.0 forcing output\n")
        f.write(f"# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")

        if unit:
            f.write(f"# Unit ID: {unit.id}\n")
            f.write(f"# Location: ({unit.x:.6f}, {unit.y:.6f})\n")
            f.write(f"# Elevation: {unit.elevation:.1f} m\n")
            f.write(f"# Surface type: {unit.surface_type}\n")

        if variables:
            f.write("# Variables:\n")
            for var in variables:
                info = self._get_variable_info(var)
                f.write(f"#   {var}: {info.get('long_name', var)} [{info.get('units', '?')}]\n")

        f.write("#\n")

    def _dataset_to_dataframe(
        self,
        ds: xr.Dataset,
        unit_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert xarray Dataset to pandas DataFrame.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset (single unit, no unit_id dimension).
        unit_id : str, optional
            Unit ID to add as column (for long format).

        Returns
        -------
        pd.DataFrame
            Tabular data.
        """
        # Select variables
        if self.variables:
            vars_to_use = [v for v in self.variables if v in ds.data_vars]
        else:
            vars_to_use = list(ds.data_vars)

        # Convert to DataFrame
        df = ds[vars_to_use].to_dataframe()

        # Reset index to get time as column
        df = df.reset_index()

        # Drop unit_id column if it was added by reset_index (from multi-unit dataset)
        if "unit_id" in df.columns and unit_id is not None:
            df = df.drop(columns=["unit_id"])

        # Add unit_id if requested
        if unit_id is not None:
            df.insert(1, "unit_id", unit_id)

        return df

    def _write_single_unit(
        self,
        forcing: xr.Dataset,
        unit: SpatialUnit,
        output_dir: Path,
        unit_idx: Optional[int] = None,
    ) -> Path:
        """Write CSV for a single unit.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data.
        unit : SpatialUnit
            Spatial unit.
        output_dir : Path
            Output directory.
        unit_idx : int, optional
            Index in unit_id dimension.

        Returns
        -------
        Path
            Written file path.
        """
        # Extract data for this unit
        if unit_idx is not None and "unit_id" in forcing.dims:
            ds = forcing.isel(unit_id=unit_idx)
        else:
            ds = forcing

        # Convert to DataFrame
        df = self._dataset_to_dataframe(ds)

        # Format time column
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime(self.date_format)

        filepath = output_dir / f"forcing_{unit.id}.csv"

        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            if self.include_metadata_header:
                variables = self.variables or list(ds.data_vars)
                self._write_metadata_header(f, unit, variables)

        # Append data (using pandas to_csv with mode='a')
        df.to_csv(
            filepath,
            mode="a",
            index=False,
            float_format=self.float_format,
            na_rep=self.na_rep,
        )

        return filepath

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write forcing data to CSV files.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time,) or (time, unit_id).
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.single_file and len(units) > 1:
            if self.wide_format:
                return self._write_wide_combined(forcing, units, output_dir)
            else:
                return self._write_long_combined(forcing, units, output_dir)
        else:
            # One file per unit
            written = []
            for i, unit in enumerate(units):
                filepath = self._write_single_unit(forcing, unit, output_dir, i)
                written.append(filepath)
            return written

    def _write_long_combined(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write all units to a single file in long format.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data.
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file path.
        """
        dfs = []
        for i, unit in enumerate(units):
            if "unit_id" in forcing.dims:
                ds = forcing.isel(unit_id=i)
            else:
                ds = forcing

            df = self._dataset_to_dataframe(ds, unit_id=unit.id)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Format time column
        if "time" in combined.columns:
            combined["time"] = pd.to_datetime(combined["time"]).dt.strftime(self.date_format)

        filepath = output_dir / "forcing_combined.csv"

        # Write with metadata header
        with open(filepath, "w", encoding="utf-8") as f:
            if self.include_metadata_header:
                variables = self.variables or list(forcing.data_vars)
                self._write_metadata_header(f, None, variables)
                f.write(f"# Units: {', '.join(u.id for u in units)}\n")
                f.write("#\n")

        combined.to_csv(
            filepath,
            mode="a",
            index=False,
            float_format=self.float_format,
            na_rep=self.na_rep,
        )

        return [filepath]

    def _write_wide_combined(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write all units to a single file in wide format.

        One column per unit per variable: temp_unit1, temp_unit2, ...

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data.
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file path.
        """
        # Select variables
        if self.variables:
            vars_to_use = [v for v in self.variables if v in forcing.data_vars]
        else:
            vars_to_use = list(forcing.data_vars)

        # Build wide DataFrame
        if "unit_id" not in forcing.dims:
            # Single unit case
            df = self._dataset_to_dataframe(forcing)
        else:
            # Multi-unit case
            times = pd.to_datetime(forcing.time.values)
            df = pd.DataFrame({"time": times})

            for var in vars_to_use:
                for i, unit in enumerate(units):
                    col_name = f"{var}_{unit.id}"
                    df[col_name] = forcing[var].isel(unit_id=i).values

        # Format time column
        df["time"] = df["time"].dt.strftime(self.date_format)

        filepath = output_dir / "forcing_wide.csv"

        # Write with metadata header
        with open(filepath, "w", encoding="utf-8") as f:
            if self.include_metadata_header:
                self._write_metadata_header(f, None, vars_to_use)
                f.write(f"# Units: {', '.join(u.id for u in units)}\n")
                f.write("# Format: wide (one column per unit per variable)\n")
                f.write("#\n")

        df.to_csv(
            filepath,
            mode="a",
            index=False,
            float_format=self.float_format,
            na_rep=self.na_rep,
        )

        return [filepath]

    @staticmethod
    def read(
        filepath: Path,
        comment: str = "#",
        parse_dates: list[str] = None,
    ) -> pd.DataFrame:
        """Read a CSV forcing file back to DataFrame.

        Parameters
        ----------
        filepath : Path
            CSV file path.
        comment : str
            Comment character for header lines.
        parse_dates : list[str], optional
            Columns to parse as dates. Defaults to ["time"].

        Returns
        -------
        pd.DataFrame
            Loaded data.
        """
        if parse_dates is None:
            parse_dates = ["time"]

        return pd.read_csv(
            filepath,
            comment=comment,
            parse_dates=parse_dates,
        )
