from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from openpyxl import load_workbook

from cde_excel import write_single_sheet_workbook
from scripts.extractor import make_qr_png, qr_payload_to_plain_text


def test_delivery_qr_text_matches_report_fields():
    payload = {
        "document_type": "Data Delivery Report",
        "reference_no": "CD533/620/01",
        "request_no": "260719000001",
        "date": "19 July, 2026",
        "customer_name": "Example Customer",
        "customer_address": "Dodoma",
        "customer_phone": "+255700000000",
        "customer_email": "customer@example.com",
        "element": "CHIRPS Precipitation",
        "data_type": "Annual Data",
        "station_name": "Dodoma",
        "period": "1991-01-01 to 2025-12-31",
        "mode_of_delivery": "Electronic copy",
        "served_by": "TMA Officer",
        "file_name": "data.xlsx",
        "verification_url": "https://example.test/verify/CDE-1",
    }
    text = qr_payload_to_plain_text(payload)
    for expected in (
        "Ref. No.: CD533/620/01",
        "Request No. (yymmno): 260719000001",
        "Parameter(s) provided: CHIRPS Precipitation - Annual Data",
        "Station(s) provided: Dodoma",
        "Mode of Delivery: Electronic copy",
        "Attended by: TMA Officer",
    ):
        assert expected in text
    assert "\n\n" not in text


def test_simple_excel_has_no_dropdowns_compact_columns_and_full_qr():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "output.xlsx"
        frame = pd.DataFrame({"Year": [2024, 2025], "Precipitation (mm)": [123.4, 145.6]})
        payload = {
            "document_type": "Data Delivery Report",
            "reference_no": "CD533/620/01",
            "request_no": "260719000001",
            "element": "CHIRPS Precipitation",
            "data_type": "Annual Data",
            "station_name": "Dodoma",
            "period": "2024-01-01 to 2025-12-31",
            "served_by": "TMA Officer",
        }
        write_single_sheet_workbook(path, [("Annual Data", frame)], qr_payload=payload)
        wb = load_workbook(path)
        ws = wb["Data"]
        assert ws.auto_filter.ref is None
        assert ws.freeze_panes is None
        assert ws.column_dimensions["A"].width <= 24
        assert ws.column_dimensions["B"].width <= 24
        assert len(ws._images) == 1
        wb.close()


def test_full_qr_png_is_generated_at_scan_friendly_resolution():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "qr.png"
        make_qr_png({
            "document_type": "Proforma Invoice",
            "reference_no": "CDE-PFI-20260719-0001",
            "date": "19 July, 2026",
            "customer_name": "Example Customer",
            "customer_address": "Dodoma, Tanzania",
            "customer_category": "Government",
            "data_type": "Monthly",
            "stations": 2,
            "parameters": 8,
            "years": 10,
            "description": "Monthly meteorological and climate data service generated from CDE.",
            "total_fee": "TZS 1,250,000",
            "served_by": "TMA Officer",
        }, path)
        from PIL import Image
        image = Image.open(path)
        assert image.width >= 500
        assert image.height == image.width
