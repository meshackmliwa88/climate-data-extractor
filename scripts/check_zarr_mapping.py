#!/usr/bin/env python3
"""Validate CDE catalog mappings against the deployed Zarr hierarchy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cde_store import default_data_dir  # noqa: E402
from scripts.extractor import list_available_sources, load_catalog  # noqa: E402

CATALOG = PROJECT_ROOT / "config" / "zarr_catalog.json"


def main() -> int:
    data_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else default_data_dir(PROJECT_ROOT)
    catalog_path = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else CATALOG
    catalog = load_catalog(catalog_path)
    status = list_available_sources(catalog, data_dir)

    print(f"Catalog: {catalog_path}")
    print(f"Zarr root: {data_dir}")
    print()
    missing = 0
    found = 0
    for key, item in status.items():
        print(f"[{key}] {item['label']}")
        for resolution, info in item["available_by_frequency"].items():
            if info.get("season_file_counts"):
                details = ", ".join(
                    f"{season}:{'OK' if count else 'MISS'}"
                    for season, count in info["season_file_counts"].items()
                )
                print(f"  {resolution:<8} {details}")
                found += sum(1 for count in info["season_file_counts"].values() if count)
                missing += sum(1 for count in info["season_file_counts"].values() if not count)
            else:
                ok = bool(info.get("available"))
                print(f"  {resolution:<8} {'OK' if ok else 'MISS'}  {info.get('pattern', '')}")
                found += int(ok)
                missing += int(not ok)
        print()

    climate_index_dir = data_dir / "climate_indices"
    climate_index_count = len(list(climate_index_dir.glob("*.zarr"))) if climate_index_dir.exists() else 0
    print(f"Mapped catalog entries found: {found}")
    print(f"Missing catalog entries: {missing}")
    print(f"Precomputed climate-index stores: {climate_index_count}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
