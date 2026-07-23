"""Shared Zarr-only data-store helpers for CDE.

Every operational data read is restricted to a ``.zarr`` directory.  Stores
are opened lazily with their encoded chunks so requests read only the selected
variable, time range and grid location.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask
    dask.config.set(
        scheduler="threads",
        num_workers=max(1, int(os.environ.get("CDE_DASK_WORKERS", "2"))),
        **{"array.slicing.split-large-chunks": False},
    )
except Exception:
    pass

ZARR_SUFFIX = ".zarr"


_TIME_CANDIDATES = ("time", "valid_time", "datetime", "date", "forecast_time")


def _datetime_like(values) -> bool:
    """Return True when an array contains usable date/time values."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return True
    if array.size == 0:
        return False
    # cftime values are intentionally detected without importing cftime.
    first = next((value for value in array.reshape(-1)[:8] if value is not None), None)
    if first is not None and first.__class__.__module__.startswith("cftime"):
        return True
    if array.dtype.kind in {"O", "U", "S"}:
        parsed = pd.to_datetime(pd.Series(array.reshape(-1)[:64]), errors="coerce")
        return bool(parsed.notna().mean() >= 0.75)
    return False


def resolve_time_axis(obj: xr.Dataset | xr.DataArray, preferred: str | None = None) -> tuple[str, str]:
    """Resolve the date coordinate and its underlying dimension.

    Operational Zarr stores may expose ``valid_time`` as a one-dimensional
    coordinate on a dimension named ``time``; some older stores keep it as a
    data variable.  Returning both names lets callers use positional masking
    without requiring an xarray index.
    """
    candidates = []
    for name in (preferred, *_TIME_CANDIDATES):
        if name and name not in candidates:
            candidates.append(name)

    def variable(name: str):
        if name in obj.coords:
            return obj.coords[name]
        if isinstance(obj, xr.Dataset) and name in obj.variables:
            return obj[name]
        return None

    # Prefer explicitly named, datetime-like one-dimensional coordinates.
    for name in candidates:
        value = variable(name)
        if value is not None and value.ndim == 1 and value.dims:
            units = str(value.attrs.get("units", "")).lower()
            standard = str(value.attrs.get("standard_name", "")).lower()
            if _datetime_like(value.values) or "since" in units or "time" in standard:
                return name, value.dims[0]

    # Search every one-dimensional coordinate/data variable for dates.
    names = list(obj.coords)
    if isinstance(obj, xr.Dataset):
        names.extend(name for name in obj.variables if name not in names)
    for name in names:
        value = variable(name)
        if value is not None and value.ndim == 1 and value.dims and _datetime_like(value.values):
            return str(name), value.dims[0]

    # Last resort: a named dimension coordinate. This supports numeric dates
    # when xarray decoded them imperfectly, while date slicing will still emit
    # a clear error if no actual date values exist.
    for name in candidates:
        if name in obj.dims:
            return name, name
    raise ValueError(
        "No usable time coordinate was found. Expected one of: "
        + ", ".join(candidates)
    )


def normalise_time_coordinate(ds: xr.Dataset, preferred: str = "time") -> xr.Dataset:
    """Promote and standardise a store's date coordinate lazily."""
    name, dim = resolve_time_axis(ds, preferred)
    out = ds
    if name in out.data_vars:
        out = out.set_coords(name)
    if name != dim and name in out.coords and out[name].dims == (dim,):
        try:
            out = out.swap_dims({dim: name})
            dim = name
        except Exception:
            pass
    # Use one canonical dimension across stores so merge/concat cannot create
    # an unindexed integer ``time`` dimension beside the actual date values.
    if dim != preferred and dim in out.dims and preferred not in out.variables:
        out = out.rename({dim: preferred})
        dim = preferred
    elif dim != preferred and dim in out.dims and preferred in out.coords:
        # The preferred coordinate already exists on this dimension. Swap to it.
        try:
            out = out.swap_dims({dim: preferred})
            dim = preferred
        except Exception:
            pass
    if preferred in out.coords:
        try:
            out = out.sortby(preferred)
        except Exception:
            pass
    return out


def _datetime_mask(values, start=None, end=None) -> np.ndarray | None:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return np.zeros(0, dtype=bool)
    if np.issubdtype(array.dtype, np.datetime64) or array.dtype.kind in {"O", "U", "S"}:
        parsed = pd.to_datetime(pd.Series(array), errors="coerce")
        if parsed.notna().any():
            # Pandas may expose a read-only NumPy view here (notably with
            # newer pandas/Python builds).  Always allocate a writable boolean
            # array before applying the date bounds in-place.
            mask = np.asarray(parsed.notna().to_numpy(), dtype=bool).copy()
            if start not in (None, ""):
                start_mask = np.asarray(
                    (parsed >= pd.Timestamp(start)).to_numpy(), dtype=bool
                )
                mask = np.logical_and(mask, start_mask)
            if end not in (None, ""):
                end_mask = np.asarray(
                    (parsed <= pd.Timestamp(end)).to_numpy(), dtype=bool
                )
                mask = np.logical_and(mask, end_mask)
            return np.asarray(mask, dtype=bool)
    return None


def slice_time_range(
    obj: xr.Dataset | xr.DataArray,
    preferred: str | None,
    start=None,
    end=None,
) -> tuple[xr.Dataset | xr.DataArray, str]:
    """Slice a Dataset/DataArray by dates even when time is not an xarray index.

    The previous direct ``.sel(time=slice('YYYY', 'YYYY'))`` call fails with
    ``TypeError: 'str' object cannot be interpreted as an integer`` whenever a
    store has a ``time`` dimension but its actual dates are held in a separate
    one-dimensional coordinate.  This helper uses a boolean positional mask,
    so it works for indexed and non-indexed operational stores alike.
    """
    name, dim = resolve_time_axis(obj, preferred)
    value = obj[name] if name in obj.coords or name in obj.variables else None
    if value is not None:
        mask = _datetime_mask(value.values, start, end)
        if mask is not None:
            positions = np.flatnonzero(mask)
            return obj.isel({dim: positions}), name

    # cftime and other xarray-native indexes are handled by a promoted index.
    candidate = obj
    if isinstance(candidate, xr.Dataset) and name in candidate.data_vars:
        candidate = candidate.set_coords(name)
    if name != dim and name in candidate.coords and candidate[name].dims == (dim,):
        try:
            candidate = candidate.swap_dims({dim: name})
            dim = name
        except Exception:
            pass
    if name in candidate.coords:
        try:
            return candidate.sel({name: slice(start, end)}), name
        except Exception as exc:
            raise ValueError(
                f"The store contains a time axis '{name}', but its values could not be filtered as dates. "
                "Rebuild the Zarr store with decoded datetime coordinates."
            ) from exc
    raise ValueError(
        f"The selected store has a '{dim}' dimension but no usable datetime coordinate. "
        "Rebuild the Zarr store while preserving time/valid_time values."
    )


def select_nearest_time(
    obj: xr.Dataset | xr.DataArray,
    preferred: str | None,
    target,
) -> tuple[xr.Dataset | xr.DataArray, str]:
    """Select the nearest timestamp without requiring an xarray index."""
    name, dim = resolve_time_axis(obj, preferred)
    value = obj[name] if name in obj.coords or name in obj.variables else None
    if value is not None:
        array = np.asarray(value.values).reshape(-1)
        parsed = pd.to_datetime(pd.Series(array), errors="coerce")
        if parsed.notna().any():
            target_ts = pd.Timestamp(target)
            valid_positions = np.flatnonzero(parsed.notna().to_numpy())
            deltas = np.abs((parsed.iloc[valid_positions] - target_ts).to_numpy(dtype="timedelta64[ns]").astype("int64"))
            position = int(valid_positions[int(np.argmin(deltas))])
            return obj.isel({dim: position}), name
    candidate = obj
    if isinstance(candidate, xr.Dataset) and name in candidate.data_vars:
        candidate = candidate.set_coords(name)
    if name != dim and name in candidate.coords and candidate[name].dims == (dim,):
        try:
            candidate = candidate.swap_dims({dim: name})
        except Exception:
            pass
    try:
        return candidate.sel({name: target}, method="nearest"), name
    except Exception as exc:
        raise ValueError(f"Unable to select the nearest date from time coordinate '{name}'.") from exc


def _drop_duplicate_times(ds: xr.Dataset, time_coord: str) -> xr.Dataset:
    if time_coord not in ds.coords:
        return ds
    try:
        values = np.asarray(ds[time_coord].values).reshape(-1)
        index = pd.Index(values)
        keep = ~index.duplicated(keep="first")
        if not bool(np.all(keep)):
            ds = ds.isel({ds[time_coord].dims[0]: np.flatnonzero(keep)})
        return ds.sortby(time_coord)
    except Exception:
        return ds

# Remember whether each store has consolidated metadata.  This avoids a failed
# metadata probe on every subsequent open while still falling back safely for
# older stores.
_OPEN_MODE_CACHE: dict[str, bool] = {}
_OPEN_MODE_LOCK = threading.Lock()


def default_data_dir(project_root: Path) -> Path:
    """Return the configured Zarr root, defaulting to ``storage/zarr``."""
    configured = os.environ.get("CDE_ZARR_DIR") or os.environ.get("CDE_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else project_root / "storage" / "zarr"


def is_zarr_store(path: str | Path) -> bool:
    return str(path).rstrip("/").lower().endswith(ZARR_SUFFIX)


def store_kind(path: str | Path) -> str:
    if not is_zarr_store(path):
        raise ValueError(f"The selected path is not a supported climate data store: {path}")
    return "Zarr"


def store_display_name(path: str | Path) -> str:
    return Path(str(path).rstrip("/")).name


@lru_cache(maxsize=64)
def _cached_store_inventory(root_text: str, version: int) -> tuple[str, ...]:
    """Cache store paths so a page request does not repeatedly scan all chunks."""
    root = Path(root_text)
    if not root.exists():
        return ()
    return tuple(str(path) for path in sorted(root.rglob("*.zarr")) if path.is_dir())


def _inventory_version(root: Path) -> int:
    """Cheaply detect newly added or removed resolution/store directories."""
    if not root.exists():
        return 0
    stamps = [root.stat().st_mtime_ns]
    try:
        stamps.extend(child.stat().st_mtime_ns for child in root.iterdir() if child.is_dir())
    except OSError:
        pass
    return hash(tuple(stamps))


def clear_store_inventory_cache() -> None:
    _cached_store_inventory.cache_clear()


def iter_data_stores(data_dir: str | Path) -> Iterator[Path]:
    """Yield Zarr stores only."""
    root = Path(data_dir).expanduser().resolve()
    for value in _cached_store_inventory(str(root), _inventory_version(root)):
        yield Path(value)


def _has_dask() -> bool:
    return importlib.util.find_spec("dask") is not None


def _default_chunks():
    # {} tells xarray to preserve the chunks encoded in the Zarr arrays.
    return {} if _has_dask() else None


def _validate_zarr_path(path: Path) -> None:
    if not is_zarr_store(path):
        raise ValueError(f"The selected path is not a supported climate data store: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Climate data store not found: {path}")


def open_data_store(
    path: str | Path,
    *,
    time_coord: str | None = None,
    chunks=None,
    decode_times: bool = True,
) -> xr.Dataset:
    """Open one Zarr store lazily without loading complete arrays."""
    zarr_path = Path(path)
    _validate_zarr_path(zarr_path)
    if chunks == "eager":
        # Explicit eager mode used only as a compatibility retry for malformed
        # or non-standard operational chunk metadata.
        use_chunks = None
    else:
        use_chunks = _default_chunks() if chunks is None else chunks
    last_error: Exception | None = None
    cache_key = str(zarr_path.resolve())
    with _OPEN_MODE_LOCK:
        preferred = _OPEN_MODE_CACHE.get(cache_key)
    modes = (preferred,) if preferred is not None else (True, False)
    if preferred is not None:
        modes = (preferred, not preferred)
    for consolidated in modes:
        try:
            dataset = xr.open_zarr(
                str(zarr_path),
                consolidated=consolidated,
                chunks=use_chunks,
                decode_times=decode_times,
            )
            with _OPEN_MODE_LOCK:
                _OPEN_MODE_CACHE[cache_key] = bool(consolidated)
            if time_coord:
                try:
                    dataset = normalise_time_coordinate(dataset, time_coord)
                except ValueError:
                    # Non-temporal helper stores are still allowed; consumers
                    # that require time will raise a targeted error later.
                    pass
            return dataset
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to open climate data store {zarr_path}: {last_error}") from last_error


def open_data_stores(
    paths: Sequence[str | Path],
    *,
    time_coord: str = "time",
    chunks=None,
    decode_times: bool = True,
) -> xr.Dataset:
    """Open and combine one or more Zarr stores lazily."""
    normalized = [Path(p) for p in paths]
    if not normalized:
        raise FileNotFoundError("No climate data stores were supplied.")
    for path in normalized:
        _validate_zarr_path(path)
    if len(normalized) == 1:
        return open_data_store(normalized[0], time_coord=time_coord, chunks=chunks, decode_times=decode_times)

    datasets = [open_data_store(p, time_coord=time_coord, chunks=chunks, decode_times=decode_times) for p in normalized]
    datasets = [normalise_time_coordinate(ds, time_coord) for ds in datasets]

    errors: list[Exception] = []
    # 1) Coordinate-aware combination handles sequential stores of the same
    # variable without loading their arrays. Explicit join removes the xarray
    # FutureWarning seen in the production journal.
    try:
        combined = xr.combine_by_coords(
            datasets,
            combine_attrs="override",
            join="outer",
        )
        return _drop_duplicate_times(combined, time_coord)
    except Exception as exc:
        errors.append(exc)

    # 2) Prepared temperature products are sometimes split into mean/min/max
    # stores covering identical dates. These must be merged as variables, not
    # concatenated into a new integer time dimension.
    try:
        combined = xr.merge(
            datasets,
            compat="no_conflicts",
            join="outer",
            combine_attrs="override",
        )
        return _drop_duplicate_times(combined, time_coord)
    except Exception as exc:
        errors.append(exc)

    # 3) Final fallback for sequential stores with overlapping metadata.
    try:
        combined = xr.concat(
            datasets,
            dim=time_coord,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="override",
        )
        return _drop_duplicate_times(combined, time_coord)
    except Exception as exc:
        errors.append(exc)
        detail = "; ".join(str(error) for error in errors[-3:])
        raise RuntimeError(f"Unable to combine Zarr stores along '{time_coord}': {detail}") from exc


def glob_store_paths(pattern: str | Path) -> list[str]:
    """Return only existing Zarr directories matched by a path or glob."""
    text = str(pattern)
    matches = glob.glob(text, recursive=True) if any(ch in text for ch in "*?[") else [text]
    return sorted({str(Path(p)) for p in matches if Path(p).is_dir() and is_zarr_store(p)})


def catalog_pattern_variants(pattern_name: str) -> list[str]:
    """Normalize one catalog entry to a Zarr-only pattern."""
    pattern = str(pattern_name or "").strip()
    if not pattern:
        return []
    return [pattern if pattern.lower().endswith(ZARR_SUFFIX) else pattern + ZARR_SUFFIX]


def close_datasets(datasets: Iterable[xr.Dataset]) -> None:
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass
