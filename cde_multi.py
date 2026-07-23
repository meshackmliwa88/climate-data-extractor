"""Lightweight multi-selection extraction helpers for the unified CDE workspace.

The module opens only the explicitly requested point slices. It writes one
simple worksheet per weather-element/resolution selection and
releases each frame before continuing to the next selection.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xlsxwriter

from cde_products import DATASETS, extract_point_series, product_data_cache, slugify, y_axis_label
from scripts.extractor import default_download_context, make_qr_png, style_excel


VARIABLE_OPTIONS: dict[str, list[dict[str, str]]] = {
    "chirps_rainfall": [
        {"value": "precip", "label": "CHIRPS Precipitation"},
    ],
    "era5_total_precipitation": [
        {"value": "tp", "label": "ERA5 Precipitation"},
    ],
    "era5_temperature": [
        {"value": "ta", "label": "Mean 2m Temperature"},
    ],
    "era5_temperature_stats": [
        {"value": "ta", "label": "Mean Temperature"},
        {"value": "tmin", "label": "Minimum Temperature"},
        {"value": "tmax", "label": "Maximum Temperature"},
    ],
    "era5_dew_point": [
        {"value": "d2m", "label": "Dew Point Temperature at 2 m"},
    ],
    "era5_relative_humidity": [
        {"value": "r", "label": "Relative Humidity"},
    ],
    "era5_skin_temperature": [
        {"value": "skt", "label": "Skin Temperature"},
    ],
    "era5_soil_temperature": [
        {"value": "stl1", "label": "Soil Temperature Level 1"},
    ],
    "era5_soil_water": [
        {"value": "swvl1", "label": "Volumetric Soil Moisture"},
    ],
    "era5_wind": [
        {"value": "wind_speed", "label": "Wind Speed"},
        {"value": "wind_direction", "label": "Wind Direction"},
    ],
    "era5_pressure_cloud": [
        {"value": "sp", "label": "Surface Pressure"},
        {"value": "tcc", "label": "Total Cloud Cover"},
    ],
}

RESOLUTION_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "monthly": "Monthly",
    "annual": "Annual",
    "seasonal": "Seasonal",
}


def variable_label(dataset: str, variable: str) -> str:
    for item in VARIABLE_OPTIONS.get(dataset, []):
        if item["value"] == variable:
            return item["label"]
    return str(variable).replace("_", " ").title()


def selected_season_jobs(resolutions: Iterable[str], seasons: Iterable[str]) -> list[tuple[str, str | None]]:
    jobs: list[tuple[str, str | None]] = []
    selected_seasons = [str(s).strip().upper() for s in seasons if str(s).strip()]
    for resolution in resolutions:
        resolution = str(resolution).strip().lower()
        if not resolution:
            continue
        if resolution == "seasonal":
            for season in selected_seasons or ["MAM"]:
                jobs.append((resolution, season))
        else:
            jobs.append((resolution, None))
    return jobs


def _zero_decimal(label: str) -> bool:
    text = str(label).lower()
    return any(token in text for token in ("relative humidity", "wind speed", "wind direction"))


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _merge_variables_for_job(
    *,
    data_dir: Path,
    dataset: str,
    resolution: str,
    season: str | None,
    variables: list[str],
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    merged: pd.DataFrame | None = None
    contexts: dict[str, dict[str, Any]] = {}
    for variable in variables:
        frame, context = extract_point_series(
            data_dir,
            dataset,
            resolution,
            latitude,
            longitude,
            start_date,
            end_date,
            variable=variable,
            season=season,
        )
        contexts[variable] = context
        column = variable_label(dataset, variable)
        part = frame.rename(columns={"value": column})[["time", column]]
        merged = part if merged is None else merged.merge(part, on="time", how="outer", sort=True)
    if merged is None:
        merged = pd.DataFrame(columns=["time"])
    return merged.sort_values("time").reset_index(drop=True), contexts


def _safe_sheet_name(label: str, used: set[str]) -> str:
    import re

    base = re.sub(r"[\\/*?:\[\]]+", " ", str(label or "Data"))
    base = re.sub(r"\s+", " ", base).strip() or "Data"
    base = base[:31]
    candidate = base
    counter = 2
    while candidate.lower() in used:
        suffix = f" {counter}"
        candidate = f"{base[:31-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate.lower())
    return candidate


def generate_multi_extraction(
    params: dict[str, Any],
    data_dir: Path,
    export_dir: Path,
) -> dict[str, Any]:
    """Generate one workbook with a separate simple sheet per selection.

    A selection is one weather element × temporal resolution × season. This
    avoids mixing unlike time scales in one table and keeps every requested
    output easy to inspect, filter and print.
    """
    dataset = str(params.get("dataset") or "chirps_rainfall")
    if dataset not in DATASETS:
        raise ValueError("Select a valid dataset.")
    available_variables = {item["value"] for item in VARIABLE_OPTIONS.get(dataset, [])}
    variables = [str(v) for v in params.get("variables", []) if str(v) in available_variables]
    if not variables:
        raise ValueError("Select at least one weather element.")
    available_resolutions = set(DATASETS[dataset].get("resolutions", {}))
    resolutions = [str(r) for r in params.get("resolutions", []) if str(r) in available_resolutions]
    if not resolutions:
        raise ValueError("Select at least one temporal resolution.")
    jobs = selected_season_jobs(resolutions, params.get("seasons", []))
    if not jobs:
        raise ValueError("Select at least one temporal resolution.")

    latitude = float(params.get("latitude"))
    longitude = float(params.get("longitude"))
    location = str(params.get("location_name") or "Selected Location")
    start_date = str(params.get("start_date") or "")
    end_date = str(params.get("end_date") or "")
    if not start_date or not end_date or start_date > end_date:
        raise ValueError("Provide a valid start and end period.")

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CDE_Data_Extraction_{slugify(DATASETS[dataset]['label'])}_{slugify(location)}_{stamp}.xlsx"
    output = export_dir / filename

    selection_count = len(variables) * len(jobs)
    qr_payload = default_download_context(output)
    qr_payload.update({
        "document_type": "Data Delivery Report",
        "reference_no": "CD533/620/01",
        "request_no": str(qr_payload.get("download_id") or "").replace("CDE-", "")[:12],
        "mode_of_delivery": "Electronic copy",
        "file_name": filename,
        "source": DATASETS[dataset]["label"],
        "element": ", ".join(variable_label(dataset, v) for v in variables),
        "data_type": ", ".join(RESOLUTION_LABELS.get(r, r.title()) for r in resolutions),
        "period": f"{start_date} to {end_date}",
        "station_name": location,
        "latitude": latitude,
        "longitude": longitude,
        "description": "Each selected element, resolution and season is stored in a separate worksheet.",
    })

    options = {"constant_memory": True, "strings_to_urls": False, "nan_inf_to_errors": True}
    with TemporaryDirectory() as tmpdir:
        qr_path = Path(tmpdir) / "verification.png"
        make_qr_png(qr_payload, qr_path, compact=False)
        workbook = xlsxwriter.Workbook(str(output), options)

        title_fmt = workbook.add_format({"bold": True, "font_size": 12, "valign": "vcenter"})
        section_fmt = workbook.add_format({"bold": True})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#E7E6E6", "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1})
        text_fmt = workbook.add_format({"border": 1, "valign": "top", "text_wrap": True})
        one_fmt = workbook.add_format({"border": 1, "num_format": "0.0", "valign": "top"})
        zero_fmt = workbook.add_format({"border": 1, "num_format": "0", "valign": "top"})
        coord_fmt = workbook.add_format({"border": 1, "num_format": "0.0000", "valign": "top"})
        datetime_fmt = workbook.add_format({"border": 1, "num_format": "yyyy-mm-dd hh:mm", "valign": "top"})
        date_fmt = workbook.add_format({"border": 1, "num_format": "yyyy-mm-dd", "valign": "top"})
        alt_text_fmt = text_fmt
        alt_one_fmt = one_fmt
        alt_zero_fmt = zero_fmt
        alt_coord_fmt = coord_fmt
        alt_datetime_fmt = datetime_fmt
        alt_date_fmt = date_fmt
        qr_title_fmt = workbook.add_format({"bold": True, "align": "center"})

        used_sheet_names: set[str] = set()
        total_rows = 0
        logical_rows_by_job: dict[tuple[str, str], int] = {}
        selection_number = 0
        nearest_coordinates: list[tuple[float, float]] = []
        with product_data_cache():
            for variable in variables:
                for resolution, season in jobs:
                    selection_number += 1
                    element_label = variable_label(dataset, variable)
                    short_element = element_label.replace("CHIRPS ", "").replace("ERA5 ", "")
                    sheet_label = f"{short_element} {RESOLUTION_LABELS.get(resolution, resolution.title())}"
                    if season:
                        sheet_label += f" {season}"
                    sheet_name = _safe_sheet_name("Data" if selection_count == 1 else sheet_label, used_sheet_names)

                    frame, context = extract_point_series(
                        Path(data_dir), dataset, resolution, latitude, longitude,
                        start_date, end_date, variable=variable, season=season,
                    )
                    logical_rows_by_job[(resolution, season or "")] = max(
                        logical_rows_by_job.get((resolution, season or ""), 0), len(frame)
                    )
                    grid_lat = float(context.get("nearest_latitude", latitude))
                    grid_lon = float(context.get("nearest_longitude", longitude))
                    nearest_coordinates.append((grid_lat, grid_lon))
                    unit = str(context.get("unit") or "")
                    value_header = y_axis_label(element_label, unit, resolution)

                    ws = workbook.add_worksheet(sheet_name)
                    columns = ["Date / Time", "Year", "Month", "Day"]
                    if resolution == "hourly":
                        columns.append("Hour")
                    columns += ["Temporal Resolution", "Season", value_header, "Unit", "Location", "Requested Latitude", "Requested Longitude", "Grid Latitude", "Grid Longitude"]
                    last_col = len(columns) - 1
                    ws.merge_range(0, 0, 0, last_col, f"{value_header} for {location}", title_fmt)
                    ws.set_row(0, 26)
                    qr_col = last_col + 2
                    ws.write(0, qr_col, "Scan to verify", qr_title_fmt)
                    # Keep the exact weather-element label available to legacy
                    # automated readers. The QR image visually covers this
                    # white-text cell in the worksheet.
                    hidden_qr_text_fmt = workbook.add_format({"font_color": "#FFFFFF"})
                    ws.write(1, qr_col, element_label, hidden_qr_text_fmt)
                    if qr_path.exists():
                        ws.insert_image(1, qr_col, str(qr_path), {"x_scale": 0.30, "y_scale": 0.30, "object_position": 1})
                    metadata = [
                        ("Dataset", DATASETS[dataset]["label"]),
                        ("Weather Element", element_label),
                        ("Temporal Resolution", RESOLUTION_LABELS.get(resolution, resolution.title())),
                        ("Season", season or "Not applicable"),
                        ("Period", f"{start_date} to {end_date}"),
                        ("Requested Coordinates", f"{latitude:.4f}, {longitude:.4f}"),
                        ("Nearest Native Grid", f"{grid_lat:.4f}, {grid_lon:.4f}"),
                        ("Unit", unit),
                    ]
                    meta_row = 2
                    ws.merge_range(meta_row, 0, meta_row, last_col, "Selection Information", section_fmt)
                    meta_row += 1
                    for key, value in metadata:
                        ws.merge_range(meta_row, 0, meta_row, last_col, f"{key} : {value}", text_fmt)
                        meta_row += 1
                    data_section_row = meta_row + 1
                    ws.merge_range(data_section_row, 0, data_section_row, last_col, "Extracted Data", section_fmt)
                    header_row = data_section_row + 1
                    for col, header in enumerate(columns):
                        ws.write(header_row, col, header, header_fmt)
                    data_row = header_row + 1

                    for record in frame.itertuples(index=False, name=None):
                        if data_row >= 1_048_575:
                            workbook.close()
                            raise ValueError(f"The {sheet_name} selection exceeds the Excel worksheet row capacity. Shorten the requested period.")
                        time_value = pd.Timestamp(record[0])
                        value = _clean_value(record[1])
                        alternating = (data_row - header_row) % 2 == 0
                        row_text_fmt = alt_text_fmt if alternating else text_fmt
                        row_one_fmt = alt_one_fmt if alternating else one_fmt
                        row_zero_fmt = alt_zero_fmt if alternating else zero_fmt
                        row_coord_fmt = alt_coord_fmt if alternating else coord_fmt
                        row_datetime_fmt = alt_datetime_fmt if alternating else datetime_fmt
                        row_date_fmt = alt_date_fmt if alternating else date_fmt
                        col = 0
                        ws.write_datetime(data_row, col, time_value.to_pydatetime(), row_datetime_fmt if resolution == "hourly" else row_date_fmt); col += 1
                        ws.write_number(data_row, col, int(time_value.year), row_zero_fmt); col += 1
                        ws.write_number(data_row, col, int(time_value.month), row_zero_fmt); col += 1
                        ws.write_number(data_row, col, int(time_value.day), row_zero_fmt); col += 1
                        if resolution == "hourly":
                            ws.write_number(data_row, col, int(time_value.hour), row_zero_fmt); col += 1
                        ws.write(data_row, col, RESOLUTION_LABELS.get(resolution, resolution.title()), row_text_fmt); col += 1
                        ws.write(data_row, col, season or "", row_text_fmt); col += 1
                        if value is None:
                            ws.write_blank(data_row, col, None, row_text_fmt)
                        elif _zero_decimal(element_label):
                            ws.write_number(data_row, col, int(round(float(value))), row_zero_fmt)
                        else:
                            ws.write_number(data_row, col, round(float(value), 1), row_one_fmt)
                        col += 1
                        ws.write(data_row, col, unit, row_text_fmt); col += 1
                        ws.write(data_row, col, location, row_text_fmt); col += 1
                        ws.write_number(data_row, col, latitude, row_coord_fmt); col += 1
                        ws.write_number(data_row, col, longitude, row_coord_fmt); col += 1
                        ws.write_number(data_row, col, grid_lat, row_coord_fmt); col += 1
                        ws.write_number(data_row, col, grid_lon, row_coord_fmt)
                        data_row += 1
                        total_rows += 1

                    widths = [17, 8, 8, 8] + ([8] if resolution == "hourly" else []) + [13, 10, 12, 8, 16, 11, 11, 11, 11]
                    for col, width in enumerate(widths):
                        ws.set_column(col, col, width)
                    ws.set_column(qr_col, qr_col + 1, 10)
                    ws.set_margins(left=0.3, right=0.3, top=0.5, bottom=0.4)
                    del frame

        workbook.close()
    # Final compact content-fit widths and plain headers without filter dropdowns.
    style_excel(output)


    variable_headers = [variable_label(dataset, v) for v in variables]
    logical_rows = sum(logical_rows_by_job.values())
    return {
        "excel_path": output,
        "rows": logical_rows,
        "context": {
            "dataset_label": DATASETS[dataset]["label"],
            "location": location,
            "variables": variable_headers,
            "resolutions": resolutions,
            "seasons": list(params.get("seasons", [])),
            "start_date": start_date,
            "end_date": end_date,
            "worksheet_count": selection_count,
            "separate_selection_sheets": True,
            "rows_written_across_sheets": total_rows,
        },
        "product_title": f"Data Extraction: {DATASETS[dataset]['label']} for {location}",
        "product_subtitle": f"{selection_count} separate selection sheet(s) · {start_date[:4]}–{end_date[:4]}",
        "product_description": "Each selected weather element, temporal resolution and season is arranged in its own simple worksheet.",
        "summary_cards": [
            {"label": "Time Records", "value": f"{logical_rows:,}", "note": f"{total_rows:,} rows written across all selection sheets"},
            {"label": "Selection Sheets", "value": str(selection_count), "note": "One sheet per element × resolution × season"},
            {"label": "Variables", "value": str(len(variables)), "note": ", ".join(variable_headers)},
        ],
    }
