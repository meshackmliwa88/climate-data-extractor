from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cde_store import open_data_stores, slice_time_range
from cde_variable_selection import select_requested_statistic_dimension
from scripts.extractor import get_selected_series


class ProductionLogRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_split_store(self, name: str, variable: str, value: float) -> Path:
        path = self.root / f"{name}.zarr"
        times = pd.date_range("1991-01-01", periods=6, freq="MS")
        values = np.full((6, 2, 2), value, dtype=np.float32)
        # Reproduce the production structure that caused .sel(time=slice(str))
        # to be treated as integer positional slicing: record is the dimension,
        # while valid_time is stored as a one-dimensional data variable.
        ds = xr.Dataset(
            {
                variable: (("record", "latitude", "longitude"), values),
                "valid_time": (("record",), times.to_numpy()),
            },
            coords={"latitude": [-6.25, -6.0], "longitude": [35.5, 35.75]},
        )
        ds[variable].attrs["units"] = "K"
        ds.to_zarr(path, mode="w", consolidated=True)
        return path

    def test_nonindexed_valid_time_and_split_temperature_stores(self):
        mean = self._write_split_store("temperature_mean", "temperature_mean", 298.0)
        minimum = self._write_split_store("temperature_minimum", "temperature_minimum", 291.0)
        maximum = self._write_split_store("temperature_maximum", "temperature_maximum", 305.0)

        ds = open_data_stores([mean, minimum, maximum], time_coord="valid_time", chunks="eager")
        self.addCleanup(ds.close)
        self.assertIn("valid_time", ds.coords)
        self.assertIn("temperature_minimum", ds.data_vars)
        self.assertIn("temperature_maximum", ds.data_vars)

        cfg = {
            "time_coord": "valid_time",
            "lat_coord": "latitude",
            "lon_coord": "longitude",
            "variables": {
                "tmin": {
                    "candidate_names": ["tmin", "temperature_minimum"],
                    "conversion": "auto_kelvin_to_celsius",
                }
            },
        }
        raw = get_selected_series(ds, cfg, "tmin", -6.1, 35.7, "1991-02-01", "1991-04-30")
        self.assertEqual(len(raw["series"]), 3)
        self.assertTrue(np.allclose(raw["series"].to_numpy(), 17.85, atol=0.01))

    def test_time_slice_uses_boolean_positions_without_xarray_index(self):
        times = pd.date_range("2000-01-01", periods=5, freq="D")
        da = xr.DataArray(
            np.arange(5, dtype=float),
            dims=("record",),
            coords={"valid_time": ("record", times.to_numpy())},
        )
        selected, time_name = slice_time_range(da, "valid_time", "2000-01-02", "2000-01-04")
        self.assertEqual(time_name, "valid_time")
        self.assertEqual(selected.sizes["record"], 3)
        self.assertEqual(selected.values.tolist(), [1.0, 2.0, 3.0])

    def test_numeric_statistic_order_is_inferred_from_temperature_values(self):
        # Operational stores are not guaranteed to use mean/min/max order.
        # This intentionally uses mean/max/min and numeric labels.
        values = np.zeros((2, 3, 2, 2), dtype=np.float32)
        values[:, 0, :, :] = 298.0  # mean
        values[:, 1, :, :] = 305.0  # maximum
        values[:, 2, :, :] = 291.0  # minimum
        da = xr.DataArray(
            values,
            dims=("time", "statistic", "latitude", "longitude"),
            coords={
                "time": pd.date_range("2000-01-01", periods=2, freq="MS"),
                "statistic": [0, 1, 2],
                "latitude": [-6.25, -6.0],
                "longitude": [35.5, 35.75],
            },
        )
        tmin = select_requested_statistic_dimension(
            da, "tmin", keep_dims={"time", "latitude", "longitude"}
        )
        tmax = select_requested_statistic_dimension(
            da, "tmax", keep_dims={"time", "latitude", "longitude"}
        )
        tmean = select_requested_statistic_dimension(
            da, "ta", keep_dims={"time", "latitude", "longitude"}
        )
        self.assertAlmostEqual(float(tmin.mean()), 291.0)
        self.assertAlmostEqual(float(tmax.mean()), 305.0)
        self.assertAlmostEqual(float(tmean.mean()), 298.0)


if __name__ == "__main__":
    unittest.main()
