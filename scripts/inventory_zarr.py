#!/usr/bin/env python3
"""Write a lightweight JSON inventory of deployed Zarr stores and metadata."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cde_store import default_data_dir, iter_data_stores, open_data_store  # noqa: E402


def folder_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else default_data_dir(PROJECT_ROOT)
    output = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else PROJECT_ROOT / "storage" / "exports" / "zarr_inventory.json"
    records = []
    for store in iter_data_stores(root):
        record = {
            "relative_path": str(store.relative_to(root)),
            "name": store.name,
            "group": store.parent.name,
            "size_bytes": folder_size(store),
            "variables": [],
            "coordinates": [],
            "dimensions": {},
            "status": "ok",
        }
        try:
            with open_data_store(store, chunks=None, decode_times=False) as ds:
                record["variables"] = list(ds.data_vars)
                record["coordinates"] = list(ds.coords)
                record["dimensions"] = {str(k): int(v) for k, v in ds.sizes.items()}
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        records.append(record)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zarr_root": str(root),
        "store_count": len(records),
        "stores": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Inventory written: {output}")
    print(f"Stores: {len(records)}")
    print(f"Errors: {sum(1 for row in records if row['status'] == 'error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
