from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cde_products import _international_boundary_segments, _render_tanzania_map_axis


class FullUncroppedLakeTests(unittest.TestCase):
    def test_transboundary_lake_keeps_full_geometry_but_is_cropped_by_fixed_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            hydro = data_dir / "shapefiles" / "hydrography"
            hydro.mkdir(parents=True)
            # A complete lake polygon extends west of the Tanzania test boundary.
            lake = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"name": "Test Transboundary Lake"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[28.0, -14.0], [31.0, -14.0], [31.0, -4.0], [28.0, -4.0], [28.0, -14.0]]],
                    },
                }],
            }
            (hydro / "tanzania_lakes.geojson").write_text(json.dumps(lake), encoding="utf-8")

            boundary = np.array([
                [30.0, -10.0], [40.0, -10.0], [40.0, 0.0], [30.0, 0.0], [30.0, -10.0]
            ])
            x = np.array([30.0, 40.0])
            y = np.array([-10.0, 0.0])
            values = np.array([[1.0, 2.0], [3.0, 4.0]])

            fig, ax = plt.subplots()
            _render_tanzania_map_axis(
                ax,
                x=x,
                y=y,
                values=values,
                data_dir=data_dir,
                selected_boundaries=[boundary],
                level=1,
                dataset_key="chirps",
                show_ocean=False,
                show_lakes=True,
                show_rivers=False,
                title="Test",
                show_labels=False,
                show_cartographic_elements=False,
                render_style="grid",
            )
            self.assertLessEqual(ax.get_xlim()[0], 27.92)
            self.assertGreaterEqual(ax.get_xlim()[1], 40.08)
            self.assertEqual(tuple(round(v, 6) for v in ax.get_ylim()), (-12.0, 0.0))
            lake_collections = [item for item in ax.collections if item.get_gid() == "cde-lakes"]
            self.assertTrue(lake_collections)
            self.assertTrue(lake_collections[-1].get_clip_on())
            # The source geometry remains complete even though the axes visually
            # crop everything south of 12°S.
            lake_vertices = np.vstack([path.vertices for path in lake_collections[-1].get_paths()])
            self.assertLessEqual(float(lake_vertices[:, 1].min()), -14.0)
            boundary_collections = [item for item in ax.collections if item.get_gid() == "cde-international-boundary"]
            self.assertTrue(boundary_collections)
            self.assertGreater(boundary_collections[-1].get_zorder(), lake_collections[-1].get_zorder())
            plt.close(fig)

    def test_internal_region_edge_is_not_redrawn_as_international_boundary(self):
        # Two adjacent regions form one national rectangle.  The shared edge is
        # intentionally subdivided differently to reproduce imperfect GADM
        # topology around Lake Victoria.
        west = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 0.4], [1.0, 1.0],
            [0.0, 1.0], [0.0, 0.0],
        ])
        east = np.array([
            [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0],
            [1.0, 0.7], [1.0, 0.2], [1.0, 0.0],
        ])
        segments = _international_boundary_segments([west, east])
        self.assertTrue(segments)
        # No retained segment should lie on the internal x=1 regional border.
        for segment in segments:
            self.assertFalse(np.allclose(segment[:, 0], 1.0))
        # The outer national rectangle must still be retained.
        self.assertTrue(any(np.allclose(segment[:, 0], 0.0) for segment in segments))
        self.assertTrue(any(np.allclose(segment[:, 0], 2.0) for segment in segments))



if __name__ == "__main__":
    unittest.main()
