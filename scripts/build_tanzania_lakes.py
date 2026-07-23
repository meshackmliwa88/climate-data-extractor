#!/usr/bin/env python3
"""Build tanzania_lakes.geojson from HydroLAKES using spatial overlap.

This deliberately uses bounding-box intersection rather than lake centroids or
country attributes, so transboundary lakes such as Lake Nyasa/Lake Malawi and
Lake Tanganyika are retained.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import shapefile

TANZANIA_WINDOW = (27.5, -15.0, 42.0, 1.5)  # min lon, min lat, max lon, max lat
NYASA_WINDOW = (34.15, -12.55, 35.95, -8.65)


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def reader_from_source(source: Path) -> shapefile.Reader:
    if source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        names = archive.namelist()
        shp_name = next((name for name in names if name.lower().endswith(".shp")), None)
        if not shp_name:
            raise ValueError("No .shp file was found in the HydroLAKES ZIP archive.")
        stem = shp_name[:-4]
        shx_name = next((name for name in names if name.lower() == (stem + ".shx").lower()), None)
        dbf_name = next((name for name in names if name.lower() == (stem + ".dbf").lower()), None)
        if not shx_name or not dbf_name:
            raise ValueError("The ZIP archive must contain matching .shp, .shx and .dbf files.")
        return shapefile.Reader(
            shp=io.BytesIO(archive.read(shp_name)),
            shx=io.BytesIO(archive.read(shx_name)),
            dbf=io.BytesIO(archive.read(dbf_name)),
            encoding="latin1",
        )
    return shapefile.Reader(str(source), encoding="latin1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HydroLAKES .zip or .shp source")
    parser.add_argument("output", type=Path, help="Output tanzania_lakes.geojson")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"Source not found: {args.source}")

    reader = reader_from_source(args.source)
    field_names = [field[0] for field in reader.fields[1:]]
    features: list[dict[str, Any]] = []
    nyasa_found = False

    for shape_record in reader.iterShapeRecords():
        bbox = tuple(float(value) for value in shape_record.shape.bbox)
        if not intersects(bbox, TANZANIA_WINDOW):
            continue
        properties = dict(zip(field_names, list(shape_record.record)))
        # Convert values that the JSON encoder cannot serialize directly.
        properties = {key: (value.item() if hasattr(value, "item") else value) for key, value in properties.items()}
        geometry = shape_record.shape.__geo_interface__
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})
        name_text = " ".join(str(value) for value in properties.values()).lower()
        if "nyasa" in name_text or "malawi" in name_text or intersects(bbox, NYASA_WINDOW):
            nyasa_found = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": "tanzania_lakes",
        "features": features,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(features):,} lake feature(s) to {args.output}")
    print(f"Lake Nyasa/Lake Malawi detected: {'YES' if nyasa_found else 'NO'}")
    if not nyasa_found:
        print("WARNING: the source itself may not include Lake Nyasa/Lake Malawi.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
