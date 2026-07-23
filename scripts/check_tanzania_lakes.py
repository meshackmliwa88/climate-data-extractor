#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When this file is executed directly, Python adds only the scripts directory
# to sys.path. Add the project root so cde_products.py can be imported both
# during deployment and from an activated virtual environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cde_products import _lake_polygons_and_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the operational Tanzania lake GeoJSON used by CDE maps.")
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("/var/www/html/netcdf_data_extractor/storage/zarr"))
    args = parser.parse_args()
    _, status = _lake_polygons_and_status(args.data_dir)
    print(f"Lake file: {status.get('path', '')}")
    print(f"File found: {'YES' if status.get('file_found') else 'NO'}")
    print(f"Lake polygons loaded: {status.get('polygon_count', 0):,}")
    print("Map rendering: original lake geometries; no Tanzania-boundary clipping")
    print("Map latitude display: fixed at 0 to 12°S; geometry beyond the axes is visually cropped")
    print("Boundary layering: international boundary is rendered above lakes")
    print("Regional borders: internal region lines are not redrawn above Lake Victoria")
    print(f"Lake Nyasa/Lake Malawi detected: {'YES' if status.get('lake_nyasa_detected') else 'NO'}")
    return 0 if status.get("file_found") and status.get("lake_nyasa_detected") else 2


if __name__ == "__main__":
    raise SystemExit(main())
