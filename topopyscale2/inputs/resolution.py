"""Resolution-aware scaling utilities for multi-source NWP data.

This module provides functions to compute adaptive correction intensities
based on the ratio between NWP source resolution and target downscaling
resolution. Higher resolution sources require less aggressive corrections.
"""

from typing import Optional

import numpy as np


def resolution_scaling_factor(
    source_res_m: float,
    target_res_m: float,
    method: str = "linear",
    min_factor: float = 0.1,
    max_factor: float = 1.0,
) -> float:
    """Compute scaling factor for correction intensity based on resolution ratio.

    When downscaling from a high-resolution source (e.g., 1km COSMO) to a
    100m target, less correction is needed compared to downscaling from a
    coarse source (e.g., 31km ERA5) to the same target.

    Parameters
    ----------
    source_res_m : float
        Native horizontal resolution of the source NWP data [meters].
    target_res_m : float
        Target downscaling resolution [meters].
    method : str
        Scaling method:
        - "linear": Linear scaling with resolution ratio
        - "sqrt": Square root scaling (gentler decrease)
        - "log": Logarithmic scaling
        Default is "linear".
    min_factor : float
        Minimum scaling factor (prevents overcorrection). Default 0.1.
    max_factor : float
        Maximum scaling factor. Default 1.0.

    Returns
    -------
    float
        Scaling factor in range [min_factor, max_factor].
        Higher values mean more correction is needed.

    Examples
    --------
    >>> # ERA5 (31km) to 100m target: full correction
    >>> resolution_scaling_factor(31000, 100)
    1.0

    >>> # COSMO-1 (1km) to 100m target: reduced correction
    >>> resolution_scaling_factor(1000, 100)
    0.9

    >>> # Perfect resolution match: minimal correction
    >>> resolution_scaling_factor(100, 100)
    0.1
    """
    if source_res_m <= 0 or target_res_m <= 0:
        raise ValueError("Resolutions must be positive.")

    # Resolution ratio: >1 means source is coarser than target
    ratio = source_res_m / target_res_m

    if method == "linear":
        # Linear scaling: factor = min + (max-min) * (1 - 1/ratio)
        # At ratio=1, factor = min_factor
        # At ratio->inf, factor -> max_factor
        if ratio <= 1.0:
            factor = min_factor
        else:
            factor = min_factor + (max_factor - min_factor) * (1.0 - 1.0 / ratio)

    elif method == "sqrt":
        # Square root scaling: gentler transition
        if ratio <= 1.0:
            factor = min_factor
        else:
            factor = min_factor + (max_factor - min_factor) * (1.0 - 1.0 / np.sqrt(ratio))

    elif method == "log":
        # Logarithmic scaling: rapid initial increase, then levels off
        if ratio <= 1.0:
            factor = min_factor
        else:
            # Normalize so log(1)=0, log(inf)->1
            log_ratio = np.log10(ratio)
            # Typical range: ERA5/100m = 31000/100 = 310, log10(310) ~ 2.5
            # Scale so log10(1000) maps to roughly max_factor
            normalized = min(log_ratio / 3.0, 1.0)
            factor = min_factor + (max_factor - min_factor) * normalized

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'sqrt', or 'log'.")

    return np.clip(factor, min_factor, max_factor)


def compute_correction_weights(
    source_resolutions: dict[str, float],
    target_res_m: float,
    method: str = "linear",
) -> dict[str, float]:
    """Compute correction weights for multiple sources.

    Parameters
    ----------
    source_resolutions : dict[str, float]
        Dictionary mapping source names to their resolutions [m].
    target_res_m : float
        Target downscaling resolution [m].
    method : str
        Scaling method (see resolution_scaling_factor).

    Returns
    -------
    dict[str, float]
        Dictionary mapping source names to correction weights.

    Examples
    --------
    >>> resolutions = {"era5": 31000, "hres": 9000, "cosmo": 1000}
    >>> weights = compute_correction_weights(resolutions, 100)
    >>> # ERA5 gets highest weight, COSMO gets lowest
    """
    return {
        name: resolution_scaling_factor(res, target_res_m, method=method)
        for name, res in source_resolutions.items()
    }


def estimate_effective_resolution(
    blend_weights: dict[str, float],
    source_resolutions: dict[str, float],
) -> float:
    """Estimate effective resolution of a blended dataset.

    When multiple sources are blended, the effective resolution is
    a weighted average of the source resolutions.

    Parameters
    ----------
    blend_weights : dict[str, float]
        Dictionary mapping source names to blend weights (should sum to 1).
    source_resolutions : dict[str, float]
        Dictionary mapping source names to their resolutions [m].

    Returns
    -------
    float
        Estimated effective resolution [m].
    """
    total_weight = sum(blend_weights.values())
    if total_weight == 0:
        raise ValueError("Blend weights sum to zero.")

    effective_res = 0.0
    for name, weight in blend_weights.items():
        if name in source_resolutions:
            effective_res += (weight / total_weight) * source_resolutions[name]

    return effective_res


def adaptive_lapse_rate_weight(
    source_res_m: float,
    target_res_m: float,
    terrain_complexity: Optional[float] = None,
) -> float:
    """Compute weight for local vs regional lapse rate based on resolution.

    Higher resolution sources already capture more local variability,
    so less local adjustment is needed.

    Parameters
    ----------
    source_res_m : float
        Native horizontal resolution of the source NWP data [m].
    target_res_m : float
        Target downscaling resolution [m].
    terrain_complexity : float, optional
        Normalized terrain complexity index (0-1), if available.
        Higher values increase the weight for local corrections.

    Returns
    -------
    float
        Weight for local lapse rate correction (0-1).
        Higher values mean more local correction.
    """
    base_weight = resolution_scaling_factor(
        source_res_m, target_res_m, method="sqrt"
    )

    if terrain_complexity is not None:
        # Increase weight in complex terrain
        complexity_factor = 1.0 + 0.5 * terrain_complexity
        base_weight = min(1.0, base_weight * complexity_factor)

    return base_weight


def adaptive_radiation_partitioning_weight(
    source_res_m: float,
    target_res_m: float,
) -> float:
    """Compute weight for topographic radiation correction based on resolution.

    Parameters
    ----------
    source_res_m : float
        Native horizontal resolution of the source NWP data [m].
    target_res_m : float
        Target downscaling resolution [m].

    Returns
    -------
    float
        Weight for topographic radiation partitioning (0-1).
    """
    return resolution_scaling_factor(
        source_res_m, target_res_m, method="linear"
    )


def adaptive_wind_correction_weight(
    source_res_m: float,
    target_res_m: float,
) -> float:
    """Compute weight for wind speed correction based on resolution.

    Parameters
    ----------
    source_res_m : float
        Native horizontal resolution of the source NWP data [m].
    target_res_m : float
        Target downscaling resolution [m].

    Returns
    -------
    float
        Weight for wind speed topographic correction (0-1).
    """
    return resolution_scaling_factor(
        source_res_m, target_res_m, method="sqrt"
    )


def adaptive_precipitation_gradient_weight(
    source_res_m: float,
    target_res_m: float,
) -> float:
    """Compute weight for precipitation elevation gradient based on resolution.

    Coarse models typically underestimate orographic enhancement,
    so correction weight should be higher.

    Parameters
    ----------
    source_res_m : float
        Native horizontal resolution of the source NWP data [m].
    target_res_m : float
        Target downscaling resolution [m].

    Returns
    -------
    float
        Weight for precipitation gradient correction (0-1).
    """
    return resolution_scaling_factor(
        source_res_m, target_res_m, method="linear"
    )
