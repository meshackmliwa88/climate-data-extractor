from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cde_products import (
    PLOT_FAMILIES,
    PLOT_TYPES,
    RAINFALL_INDICES,
    find_file,
    plot_title_for,
)


class CompletePlotCatalogueTests(unittest.TestCase):
    def test_complete_user_facing_plot_catalogue(self):
        available = dict(PLOT_TYPES)
        expected = {
            "time_series", "bar", "area", "multi_line",
            "monthly_climatology", "seasonal_profile", "annual_trend",
            "anomaly", "standardized_anomaly", "spatial_map",
            "spatial_std_map", "spatial_cv_map", "heatmap",
            "mean_std_band", "std_error_bars", "standard_deviation",
            "coefficient_variation", "histogram", "box", "extreme_value",
            "scatter", "wind_rose", "variability_analysis",
        }
        self.assertTrue(expected.issubset(available))

    def test_plot_families_hide_unrelated_products(self):
        self.assertIn("area", PLOT_FAMILIES["rainfall"])
        self.assertNotIn("area", PLOT_FAMILIES["temperature"])
        self.assertIn("wind_rose", PLOT_FAMILIES["wind"])
        self.assertNotIn("wind_rose", PLOT_FAMILIES["rainfall"])
        self.assertIn("scatter", PLOT_FAMILIES["temperature"])
        self.assertNotIn("scatter", PLOT_FAMILIES["humidity"])

    def test_removed_rainy_season_indices_are_not_exposed(self):
        keys = {key for key, _ in RAINFALL_INDICES}
        self.assertNotIn("rainy_season_onset", keys)
        self.assertNotIn("rainy_season_cessation", keys)
        self.assertNotIn("length_of_rainy_season", keys)

    def test_hourly_temperature_uses_only_consolidated_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = root / "hourly" / "ERA5_Tanzania_Temperature_2M_Hourly_1940_2026.zarr"
            expected.mkdir(parents=True)
            older = root / "hourly" / "ERA5_Other_Temperature_Hourly.zarr"
            older.mkdir()
            self.assertEqual(find_file(root, "era5_temperature", "hourly"), expected)

    def test_titles_name_the_weather_element(self):
        title = plot_title_for(
            "standard_deviation",
            "ERA5 Relative Humidity",
            "Dodoma",
            "monthly",
            "1991-01-01",
            "2020-12-31",
        )
        self.assertIn("Relative Humidity", title)
        self.assertIn("Standard Deviation", title)
        self.assertIn("Dodoma", title)


if __name__ == "__main__":
    unittest.main()
