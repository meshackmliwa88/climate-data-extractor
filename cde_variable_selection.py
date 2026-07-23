"""Strict weather-variable selection helpers for CDE.

The prepared ERA5 temperature stores are not fully uniform. Some expose
separate mean/minimum/maximum variables, while others expose one temperature
array with a statistic dimension. These helpers make the requested weather
element authoritative and never silently substitute minimum with maximum (or
vice versa).
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import xarray as xr


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


MEAN_ALIASES = {
    "ta", "t2m", "tmean", "mean", "average", "avg", "mean_temperature",
    "mean_2m_temperature", "temperature_mean", "temperature_2m_mean",
    "2m_temperature_mean", "tas", "temperature",
}
MIN_ALIASES = {
    "tmin", "tn", "min", "minimum", "minimum_temperature",
    "minimum_2m_temperature", "temperature_min", "temperature_minimum",
    "temperature_2m_min", "tasmin", "2m_temperature_minimum",
}
MAX_ALIASES = {
    "tmax", "tx", "max", "maximum", "maximum_temperature",
    "maximum_2m_temperature", "temperature_max", "temperature_maximum",
    "temperature_2m_max", "tasmax", "2m_temperature_maximum",
}


def requested_statistic(variable: Any) -> str:
    name = normalize_name(variable)
    if name in MIN_ALIASES or any(token in name for token in ("minimum", "_min", "min_")):
        return "minimum"
    if name in MAX_ALIASES or any(token in name for token in ("maximum", "_max", "max_")):
        return "maximum"
    return "mean"


def aliases_for(variable: Any, candidates: Iterable[Any] = ()) -> set[str]:
    stat = requested_statistic(variable)
    base = MIN_ALIASES if stat == "minimum" else MAX_ALIASES if stat == "maximum" else MEAN_ALIASES
    return {normalize_name(v) for v in [variable, *candidates] if normalize_name(v)} | set(base)


def _variable_text(name: str, da: xr.DataArray) -> str:
    parts = [name]
    for key in ("long_name", "standard_name", "short_name", "description", "parameter"):
        parts.append(da.attrs.get(key, ""))
    return normalize_name(" ".join(str(part) for part in parts))


def _has_statistic_dimension(da: xr.DataArray) -> bool:
    for dim in da.dims:
        if normalize_name(dim) in {"statistic", "statistics", "stats", "variable", "parameter"}:
            return True
        coord = da.coords.get(dim)
        if coord is None:
            continue
        try:
            values = {normalize_name(v) for v in np.asarray(coord.values).ravel()}
        except Exception:
            continue
        if values & (MEAN_ALIASES | MIN_ALIASES | MAX_ALIASES):
            return True
    return False


def choose_data_variable(
    ds: xr.Dataset,
    requested: Any,
    candidates: Iterable[Any] = (),
) -> str:
    """Return the exact data variable for the requested weather element.

    Opposite temperature statistics are explicitly penalised. If a store has
    one numeric variable with a statistic dimension, that variable is accepted
    and the statistic dimension is selected later. Ambiguous multi-variable
    stores raise an informative error instead of plotting the wrong element.
    """
    numeric = [name for name in ds.data_vars if np.issubdtype(ds[name].dtype, np.number)]
    if not numeric:
        raise ValueError("No numeric weather variable was found in the selected Zarr store.")

    aliases = aliases_for(requested, candidates)
    lower = {normalize_name(name): name for name in numeric}
    for alias in aliases:
        if alias in lower:
            return str(lower[alias])

    statistic = requested_statistic(requested)
    opposite = MAX_ALIASES if statistic == "minimum" else MIN_ALIASES if statistic == "maximum" else (MIN_ALIASES | MAX_ALIASES)
    scored: list[tuple[int, str]] = []
    for name in numeric:
        text = _variable_text(name, ds[name])
        tokens = set(text.split("_"))
        score = 0
        for alias in aliases:
            if text == alias:
                score += 1000
            elif alias and (f"_{alias}_" in f"_{text}_" or alias in text):
                score += 180
        if any(alias and (alias == text or alias in text or alias in tokens) for alias in opposite):
            score -= 1000
        if statistic == "minimum" and ("minimum" in text or "min" in tokens):
            score += 450
        elif statistic == "maximum" and ("maximum" in text or "max" in tokens):
            score += 450
        elif statistic == "mean" and ("mean" in text or "average" in text or "avg" in tokens):
            score += 350
        scored.append((score, str(name)))

    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1]

    if len(numeric) == 1 and (_has_statistic_dimension(ds[numeric[0]]) or statistic == "mean"):
        return str(numeric[0])

    requested_text = str(requested or "weather element")
    raise ValueError(
        f"Could not uniquely match '{requested_text}' in the selected Zarr store. "
        f"Available variables: {', '.join(map(str, numeric))}."
    )


def requested_element_label(dataset_key: str, variable: Any) -> str:
    key = str(dataset_key or "")
    stat = requested_statistic(variable)
    if key == "era5_temperature_stats":
        return {"mean": "Mean Temperature", "minimum": "Minimum Temperature", "maximum": "Maximum Temperature"}[stat]
    if key == "era5_temperature":
        return "Mean 2m Temperature"
    return ""


def _semantic_coordinate_index(coord: xr.DataArray, wanted: str) -> int | None:
    aliases = {
        "mean": MEAN_ALIASES,
        "minimum": MIN_ALIASES,
        "maximum": MAX_ALIASES,
    }[wanted]
    try:
        values = [normalize_name(value) for value in np.asarray(coord.values).reshape(-1)]
    except Exception:
        return None
    for index, value in enumerate(values):
        simplified = value.replace("2m_", "").replace("temperature_", "")
        if value in aliases or simplified in aliases:
            return index
    return None


def _attribute_statistic_labels(da: xr.DataArray, size: int) -> list[str]:
    """Read optional statistic labels embedded in array attributes."""
    for key in (
        "statistic_names", "statistics", "variable_names", "component_names",
        "labels", "parameters", "parameter_names",
    ):
        raw = da.attrs.get(key)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, np.ndarray)):
            values = [normalize_name(value) for value in raw]
        else:
            text = str(raw).strip()
            try:
                import json
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    values = [normalize_name(value) for value in parsed]
                else:
                    values = [normalize_name(value) for value in re.split(r"[,;|]", text) if str(value).strip()]
            except Exception:
                values = [normalize_name(value) for value in re.split(r"[,;|]", text) if str(value).strip()]
        if len(values) == size:
            return values
    return []


def _representative_statistic_values(da: xr.DataArray, dim: str) -> np.ndarray | None:
    """Estimate each statistic member from a tiny lazy sample.

    If a prepared store has numeric statistic labels (0/1/2), positional
    assumptions are unsafe because some files use mean/max/min while others use
    mean/min/max.  Temperature minimum and maximum are instead inferred from
    their representative magnitude. Only a small sample is computed.
    """
    try:
        sample = da
        for other in sample.dims:
            if other == dim:
                continue
            size = int(sample.sizes.get(other, 1))
            if size > 1:
                # Time gets a slightly larger sample; spatial dimensions need
                # only a few cells to determine the ordering reliably.
                limit = 12 if normalize_name(other) in {"time", "valid_time", "date", "datetime"} else 3
                sample = sample.isel({other: slice(0, min(size, limit))})
        reduced = sample.mean([name for name in sample.dims if name != dim], skipna=True)
        values = reduced.compute().values if hasattr(reduced.data, "compute") else reduced.values
        values = np.asarray(values, dtype=float).reshape(-1)
        return values if len(values) == int(da.sizes.get(dim, 0)) else None
    except Exception:
        return None


def select_requested_statistic_dimension(
    da: xr.DataArray,
    requested: Any,
    *,
    keep_dims: set[str] | None = None,
) -> xr.DataArray:
    """Select the requested mean/minimum/maximum member safely.

    Semantic coordinate labels are preferred. If labels are numeric or absent,
    a small representative temperature sample determines the lowest, highest
    and middle member. This removes the production failure where selecting
    Minimum Temperature displayed Maximum Temperature.
    """
    keep = {normalize_name(name) for name in (keep_dims or set())}
    wanted = requested_statistic(requested)
    out = da
    statistic_dims = {"statistic", "statistics", "stats", "variable", "parameter", "component", "measure"}
    member_dims = {"number", "member", "expver", "surface", "level"}

    for dim in list(out.dims):
        if normalize_name(dim) in keep:
            continue
        size = int(out.sizes.get(dim, 1))
        if size <= 1:
            out = out.isel({dim: 0}, drop=True)
            continue

        coord = out.coords.get(dim)
        index = _semantic_coordinate_index(coord, wanted) if coord is not None else None

        if index is None and normalize_name(dim) in statistic_dims:
            labels = _attribute_statistic_labels(out, size)
            if labels:
                aliases = {"mean": MEAN_ALIASES, "minimum": MIN_ALIASES, "maximum": MAX_ALIASES}[wanted]
                for idx, label in enumerate(labels):
                    if label in aliases or label.replace("temperature_", "") in aliases:
                        index = idx
                        break

        if index is None and normalize_name(dim) in statistic_dims and 2 <= size <= 6:
            representative = _representative_statistic_values(out, dim)
            if representative is not None and np.isfinite(representative).any():
                valid = np.where(np.isfinite(representative))[0]
                order = valid[np.argsort(representative[valid])]
                if wanted == "minimum":
                    index = int(order[0])
                elif wanted == "maximum":
                    index = int(order[-1])
                else:
                    midpoint = (float(representative[order[0]]) + float(representative[order[-1]])) / 2.0
                    index = int(valid[np.argmin(np.abs(representative[valid] - midpoint))])

        if index is not None:
            out = out.isel({dim: int(index)}, drop=True)
        elif normalize_name(dim) in member_dims:
            out = out.isel({dim: 0}, drop=True)
        elif normalize_name(dim) in statistic_dims:
            raise ValueError(
                f"Could not identify the {wanted} temperature member in dimension '{dim}'. "
                "Add semantic statistic labels (mean, minimum, maximum) to the Zarr coordinate."
            )
    return out
