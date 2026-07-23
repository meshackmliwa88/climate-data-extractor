from pathlib import Path
import unittest

import cde_products


class RequestOnlyReferenceMapTests(unittest.TestCase):
    def test_workspace_requires_explicit_generate_action(self):
        root = Path(__file__).resolve().parents[1]
        app_text = (root / "app.py").read_text(encoding="utf-8")
        template = (root / "templates" / "plots.html").read_text(encoding="utf-8")
        self.assertIn('request.form.get("generate_requested") != "1"', app_text)
        self.assertIn('name="generate_requested" value="1"', template)
        self.assertIn('No dataset is opened and no plot is generated', template)
        self.assertIn('loading="lazy"', template)
        self.assertIn('Cache-Control', app_text)

    def test_publication_rainfall_classes_match_period(self):
        annual = cde_products._spatial_publication_style(
            "chirps_rainfall", "CHIRPS Precipitation",
            {"family": "rainfall", "months": list(range(1, 13)), "variable": "precip"},
        )
        monthly = cde_products._spatial_publication_style(
            "chirps_rainfall", "CHIRPS Precipitation",
            {"family": "rainfall", "months": [1], "variable": "precip"},
        )
        self.assertEqual(annual["boundaries"].tolist()[-1], 3000.0)
        self.assertEqual(monthly["boundaries"].tolist()[-1], 450.0)
        self.assertEqual(annual["label"], "Rainfall (mm)")

    def test_reference_titles_are_operational(self):
        title = cde_products._publication_map_title(
            "CHIRPS Precipitation",
            {"family": "rainfall", "months": list(range(1, 13)), "years": list(range(1991, 2021))},
        )
        self.assertEqual(title, "Tanzania Long-Term Mean Annual Rainfall — 1991–2020")


if __name__ == "__main__":
    unittest.main()
