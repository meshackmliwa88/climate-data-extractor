#!/usr/bin/env python3
"""Climate Data Extractor – Satellite & Reanalysis Platform (CDE).

A Flask system for extracting and analysing CHIRPS and ERA5 Zarr data.
No PostgreSQL connection is required.
"""

from __future__ import annotations
from cde_weather_names import weather_full_name

import json
import os
import re
from io import BytesIO
from tempfile import TemporaryDirectory
import secrets
import sqlite3
import time
import traceback
import urllib.parse
import urllib.request
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import wraps
from threading import BoundedSemaphore
from pathlib import Path
from typing import Any, Dict, Optional

from cde_seasons import SEASONS, get_season_months, season_year_from_date, is_sum_variable
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from scripts.extractor import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_EXPORT_DIR,
    PROJECT_ROOT,
    build_single_station,
    StationPoint,
    list_available_sources,
    load_catalog,
    write_excel_output,
    make_qr_png,
    qr_payload_to_plain_text,
    data_type_label,
)

from cde_products import (
    ALL_INDICES,
    RAINFALL_INDICES,
    TEMPERATURE_INDICES,
    OTHER_INDICES,
    DATASETS as PRODUCT_DATASETS,
    PLOT_TYPES,
    SEASON_DEFINITIONS as PRODUCT_SEASONS,
    generate_indices,
    generate_plot_product,
    slugify,
    variable_display_name,
    dataset_allowed_plots,
    product_data_cache,
    INDEX_RESOLUTION_RULES,
    INDEX_RESOLUTION_LABELS,
    INDEX_PLOT_TYPES,
    INDEX_PLOT_RULES,
)

from cde_analysis import generate_analysis_bundle
from cde_excel import combine_workbooks_to_separate_sheets
from cde_multi import (
    VARIABLE_OPTIONS as MULTI_VARIABLE_OPTIONS,
    RESOLUTION_LABELS as MULTI_RESOLUTION_LABELS,
    generate_multi_extraction,
    selected_season_jobs,
    variable_label as multi_variable_label,
)

APP_NAME = "Climate Data Extractor – Satellite & Reanalysis Platform"
APP_SHORT_NAME = "CDE"

# Only one memory-intensive extraction/product request is allowed to execute at
# a time in this process.  GET requests and ordinary pages remain responsive.
_PRODUCT_REQUEST_SLOTS = BoundedSemaphore(max(1, int(os.environ.get("CDE_MAX_PRODUCT_REQUESTS", "1"))))


def single_product_request(view):
    """Prevent concurrent heavy requests from exhausting server memory."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if request.method != "POST":
            return view(*args, **kwargs)
        if not _PRODUCT_REQUEST_SLOTS.acquire(blocking=False):
            flash("Another data product is currently being prepared. Please wait for it to finish, then submit this request again.", "warning")
            return redirect(request.path)
        try:
            return view(*args, **kwargs)
        finally:
            _PRODUCT_REQUEST_SLOTS.release()
    return wrapped


def product_form_sources() -> Dict[str, Dict[str, Any]]:
    """Build the form catalogue without scanning or opening any data stores."""
    return {
        key: {
            **meta,
            "available_resolutions": list(meta.get("resolutions", {}).keys()),
            "available_plot_types": dataset_allowed_plots(key),
        }
        for key, meta in PRODUCT_DATASETS.items()
    }
LOG_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "storage" / "db"
DB_PATH = DB_DIR / "cde_app.db"
PARQUET_DB_DIR = DB_DIR / "parquet"

DATASET_DOCUMENTATION = {
    "CHIRPS": {
        "official_name": "Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS)",
        "provider": "Climate Hazards Center, University of California Santa Barbara",
        "category": "Satellite rainfall with gauge/station blending",
        "system_use": "Primary rainfall source for Tanzania point extraction where users need long historical rainfall totals.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1981–2025 in the configured Tanzania files.",
        "spatial_coverage": "Tanzania subset; coordinates are extracted by nearest grid point or interpolation logic available in the extractor.",
        "format": "Optimized multidimensional stores arranged as daily, monthly, annual and seasonal outputs.",
        "notes": [
            "Rainfall is treated as an accumulated amount in millimetres.",
            "Daily rainfall is exported directly from the daily file.",
            "Monthly, annual and seasonal products are sums of rainfall, not averages.",
            "Best suited for rainfall monitoring, climate summaries, rainfall anomalies, rainy season reports and historical station-near point analysis."
        ],
    },
    "ERA5_TEMP": {
        "official_name": "ERA5 2m temperature products",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Atmospheric reanalysis",
        "system_use": "Mean 2m temperature extraction and plotting at hourly, daily, monthly, annual and seasonal resolutions.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2026 for hourly 2m temperature and 1940–2025 for prepared longer-resolution products.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Optimized multidimensional stores arranged as hourly, daily, monthly, annual and seasonal outputs.",
        "notes": [
            "Values are treated as degrees Celsius as supplied in the prepared project files.",
            "Mean temperature is averaged for daily, monthly, annual and seasonal summaries.",
            "Minimum temperature uses minimum aggregation at daily level and mean aggregation for longer period summaries unless already provided by prepared files.",
            "Maximum temperature uses maximum aggregation at daily level and mean aggregation for longer period summaries unless already provided by prepared files."
        ],
    },
    "ERA5_TEMP_STATS": {
        "official_name": "CDE ERA5 2m temperature mean, minimum and maximum products",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Prepared atmospheric reanalysis temperature statistics",
        "system_use": "Daily, monthly, annual and seasonal mean, minimum and maximum temperature extraction, mapping, analysis and climate-index products.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2025 in CDE_ERA5_Tanzania_Temperature_Mean_Min_Max_1940_2025 Zarr stores.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Prepared Zarr stores arranged in daily, monthly, annual and seasonal folders.",
        "notes": [
            "Mean, minimum and maximum temperature are selectable as separate weather elements.",
            "Hourly extraction remains under the ERA5 2m Temperature dataset.",
            "All supported plot, map and climate-analysis products are filtered by the selected variable and resolution."
        ],
    },
    "ERA5_TP": {
        "official_name": "ERA5 total precipitation",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Atmospheric reanalysis precipitation",
        "system_use": "Alternative gridded precipitation source for rainfall/precipitation summaries and comparison with CHIRPS.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2025 in the configured Tanzania files.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Optimized multidimensional stores arranged as hourly, daily, monthly, annual and seasonal outputs.",
        "notes": [
            "The extractor automatically converts precipitation to millimetres where the file appears to be stored in metres.",
            "Daily, monthly, annual and seasonal precipitation are summed.",
            "Use this dataset when reanalysis consistency is required across long historical periods or for comparison with other ERA5 variables."
        ],
    },
    "ERA5_WIND": {
        "official_name": "ERA5 10 metre wind components and derived wind speed/direction",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Atmospheric reanalysis wind",
        "system_use": "Wind speed and wind direction extraction for operational summaries.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2025 in the configured Tanzania files.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Optimized multidimensional stores arranged as hourly, daily, monthly, annual and seasonal outputs.",
        "notes": [
            "Wind speed is reported in knots.",
            "Wind direction is reported in degrees.",
            "When U and V components are available, wind speed is derived from the vector magnitude and direction from the vector angle.",
            "Longer-period wind direction is handled as a vector mean direction rather than a simple arithmetic average."
        ],
    },
    "ERA5_RH": {
        "official_name": "ERA5 relative humidity at 1000 hPa",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Atmospheric reanalysis humidity",
        "system_use": "Relative humidity extraction for station-near operational climate reporting.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2025 in the configured Tanzania files.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Optimized multidimensional stores arranged as hourly, daily, monthly, annual and seasonal outputs.",
        "notes": [
            "The configured pressure level is 1000 hPa where the file includes pressure levels.",
            "Relative humidity is reported in percent.",
            "Daily, monthly, annual and seasonal values are averaged."
        ],
    },
    "ERA5_PRESSURE_CLOUD": {
        "official_name": "ERA5 pressure and total cloud cover",
        "provider": "ECMWF / Copernicus Climate Data Store style reanalysis product",
        "category": "Atmospheric reanalysis pressure and cloud",
        "system_use": "Mean sea level pressure, surface pressure and cloud cover extraction for weather and climate summaries.",
        "spatial_resolution": "0.25° × 0.25° grid spatial resolution.",
        "temporal_coverage": "1940–2025 in the configured Tanzania files.",
        "spatial_coverage": "Tanzania subset from the project files.",
        "format": "Optimized multidimensional stores arranged as hourly, daily, monthly, annual and seasonal outputs.",
        "notes": [
            "Pressure is automatically converted to hPa where the file appears to be stored in Pa.",
            "Total cloud cover is converted to octas where the file appears to be stored as fraction 0–1.",
            "Daily, monthly, annual and seasonal pressure/cloud values are averaged."
        ],
    },
}

app = Flask(__name__)

# Administrative/commercial modules excluded from the public build.
_REMOVED_PUBLIC_PATHS = (
    "/exports", "/cost-recovery", "/proposed-cost-recovery",
    "/customers", "/stations", "/api/usd-rate",
)

@app.before_request
def _block_removed_public_modules():
    path = request.path.rstrip("/") or "/"
    if any(path == prefix or path.startswith(prefix + "/") for prefix in _REMOVED_PUBLIC_PATHS):
        abort(404)


@app.after_request
def _cde_disable_workspace_caching(response):
    """Keep request workspaces idle and prevent browsers replaying heavy POSTs."""
    if request.path in {"/plots", "/extract"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.template_filter('weather_full_name')
def _weather_full_name_filter(value, dataset=None):
    return weather_full_name(value, dataset)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=10)
app.secret_key = os.environ.get("CDE_SECRET_KEY") or os.environ.get("NETCDF_EXTRACTOR_SECRET") or secrets.token_hex(32)


def ensure_dirs() -> None:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    ensure_dirs()
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exc: BaseException | None = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()



def generate_otp(length: int = 6) -> str:
    """Generate a numeric six-digit temporary OTP."""
    alphabet = "0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def ensure_user_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    additions = {
        "station_name": "ALTER TABLE users ADD COLUMN station_name TEXT",
        "otp": "ALTER TABLE users ADD COLUMN otp TEXT",
        "otp_generated_at": "ALTER TABLE users ADD COLUMN otp_generated_at TEXT",
        "otp_expires_at": "ALTER TABLE users ADD COLUMN otp_expires_at TEXT",
    }
    for col, sql in additions.items():
        if col not in cols:
            conn.execute(sql)



def ensure_export_log_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(export_logs)").fetchall()}
    additions = {
        "download_id": "ALTER TABLE export_logs ADD COLUMN download_id TEXT",
        "verification_url": "ALTER TABLE export_logs ADD COLUMN verification_url TEXT",
        "qr_payload": "ALTER TABLE export_logs ADD COLUMN qr_payload TEXT",
        "customer_id": "ALTER TABLE export_logs ADD COLUMN customer_id INTEGER",
        "customer_name": "ALTER TABLE export_logs ADD COLUMN customer_name TEXT",
        "customer_organization": "ALTER TABLE export_logs ADD COLUMN customer_organization TEXT",
        "customer_phone": "ALTER TABLE export_logs ADD COLUMN customer_phone TEXT",
        "customer_email": "ALTER TABLE export_logs ADD COLUMN customer_email TEXT",
        "customer_address": "ALTER TABLE export_logs ADD COLUMN customer_address TEXT",
        "cost_recovery_fee": "ALTER TABLE export_logs ADD COLUMN cost_recovery_fee TEXT",
        "customer_remarks": "ALTER TABLE export_logs ADD COLUMN customer_remarks TEXT",
        "issued_by": "ALTER TABLE export_logs ADD COLUMN issued_by TEXT",
    }
    for col, sql in additions.items():
        if col not in cols:
            conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_export_logs_download_id ON export_logs(download_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_export_logs_customer_id ON export_logs(customer_id)")


def now_eat() -> datetime:
    return datetime.now(ZoneInfo("Africa/Dar_es_Salaam"))

def init_db() -> None:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            station_name TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            password_hash TEXT NOT NULL,
            status INTEGER NOT NULL DEFAULT 1,
            force_password_change INTEGER NOT NULL DEFAULT 0,
            otp TEXT,
            otp_generated_at TEXT,
            otp_expires_at TEXT,
            last_login_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            organization TEXT,
            postal_address TEXT,
            physical_address TEXT,
            phone TEXT,
            email TEXT,
            station_location TEXT,
            requested_parameters TEXT,
            data_duration TEXT,
            cost_recovery_fee TEXT,
            remarks TEXT,
            status INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_name TEXT NOT NULL,
            region TEXT,
            district TEXT,
            station_type TEXT,
            wigos_station_identifier TEXT,
            longitude REAL,
            latitude REAL,
            status INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS export_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            customer_id INTEGER REFERENCES customers(id) ON DELETE SET NULL,
            customer_name TEXT,
            customer_organization TEXT,
            customer_phone TEXT,
            customer_email TEXT,
            customer_address TEXT,
            cost_recovery_fee TEXT,
            customer_remarks TEXT,
            issued_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            variables TEXT,
            frequencies TEXT,
            seasons TEXT,
            start_date TEXT,
            end_date TEXT,
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            output_file TEXT,
            remote_addr TEXT,
            status TEXT NOT NULL DEFAULT 'success',
            error TEXT
        );
        """
    )
    ensure_user_columns(conn)
    ensure_export_log_columns(conn)
    count = cur.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if count == 0:
        admin_email = os.environ.get("CDE_ADMIN_EMAIL", "admin@cde.local")
        admin_password = os.environ.get("CDE_ADMIN_PASSWORD") or secrets.token_urlsafe(18)
        if not os.environ.get("CDE_ADMIN_PASSWORD"):
            print(f"[CDE] Generated one-time admin password: {admin_password}")
        admin_name = os.environ.get("CDE_ADMIN_NAME", "System Administrator")
        admin_otp = os.environ.get("CDE_ADMIN_OTP") or generate_otp()
        now = datetime.now().isoformat(timespec="seconds")
        expires = (datetime.now() + timedelta(days=30)).isoformat(timespec="seconds")
        cur.execute(
            """
            INSERT INTO users (full_name, email, role, password_hash, status, force_password_change, otp, otp_generated_at, otp_expires_at)
            VALUES (?, ?, 'admin', ?, 1, 1, ?, ?, ?)
            """,
            (admin_name, admin_email, generate_password_hash(admin_password), admin_otp, now, expires),
        )
    conn.commit()
    conn.close()


def query_one(sql: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    return get_db().execute(sql, params).fetchone()


def query_all(sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(get_db().execute(sql, params).fetchall())


def row_value(row: Any, key: str, default: Any = "") -> Any:
    """Read a sqlite Row/dict value safely."""
    if row is None:
        return default
    try:
        if hasattr(row, "keys") and key in row.keys():
            value = row[key]
            return default if value is None else value
    except Exception:
        pass
    try:
        value = row.get(key, default)  # type: ignore[attr-defined]
        return default if value is None else value
    except Exception:
        return default


def customer_display_name(customer: Any) -> str:
    """Display only the registered customer name for Served By / Data Delivery Report."""
    if not customer:
        return ""
    name = str(row_value(customer, "customer_name", "")).strip()
    org = str(row_value(customer, "organization", "")).strip()
    return name or org


def customer_address_text(customer: Any) -> str:
    parts = [
        str(row_value(customer, "postal_address", "")).strip(),
        str(row_value(customer, "physical_address", "")).strip(),
    ]
    return "; ".join([p for p in parts if p])


def customer_to_dict(customer: sqlite3.Row | None) -> Dict[str, Any] | None:
    if not customer:
        return None
    return {key: customer[key] for key in customer.keys()}


def active_customers() -> list[sqlite3.Row]:
    return query_all(
        """
        SELECT * FROM customers
        WHERE status = 1
        ORDER BY id DESC
        """
    )


def get_customer(customer_id: str | int | None) -> sqlite3.Row | None:
    if not customer_id:
        return None
    try:
        cid = int(customer_id)
    except Exception:
        return None
    return query_one("SELECT * FROM customers WHERE id = ?", (cid,))


def current_user() -> Optional[sqlite3.Row]:
    uid = session.get("user_id")
    if not uid:
        return None
    return query_one("SELECT * FROM users WHERE id = ? AND status = 1", (uid,))


@app.before_request
def load_user_and_db() -> None:
    init_db()
    g.user = current_user()


@app.context_processor
def inject_globals() -> Dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "app_short_name": APP_SHORT_NAME,
        "current_user": g.get("user"),
        "csrf_token": get_csrf_token,
        "json_list_to_label": _json_list_to_label,
    }


def get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def validate_csrf() -> None:
    expected = session.get("csrf_token")
    provided = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not expected or not provided or not secrets.compare_digest(str(expected), str(provided)):
        abort(400, "Invalid or missing form security token. Please refresh the page and try again.")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.get("user"):
            return redirect(url_for("login", next=request.full_path if request.query_string else request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.get("user"):
            return redirect(url_for("login"))
        if g.user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def clean_old_exports(max_age_hours: int = 72) -> None:
    now = time.time()
    max_age = max_age_hours * 3600
    if not DEFAULT_EXPORT_DIR.exists():
        return
    for path in DEFAULT_EXPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() == ".xlsx" and (now - path.stat().st_mtime) > max_age:
            try:
                path.unlink()
            except Exception:
                pass


def log_export(row: Dict[str, Any]) -> None:
    conn = get_db()
    conn.execute(
        """
        INSERT INTO export_logs (
            user_id, customer_id, customer_name, customer_organization, customer_phone, customer_email,
            customer_address, cost_recovery_fee, customer_remarks, issued_by,
            download_id, verification_url, qr_payload, source, variables, frequencies, seasons, start_date, end_date,
            location_name, latitude, longitude, output_file, remote_addr, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("user_id"), row.get("customer_id"), row.get("customer_name"), row.get("customer_organization"),
            row.get("customer_phone"), row.get("customer_email"), row.get("customer_address"), row.get("cost_recovery_fee"),
            row.get("customer_remarks"), row.get("issued_by"), row.get("download_id"), row.get("verification_url"),
            row.get("qr_payload"), row.get("source"), json.dumps(row.get("variables", [])),
            json.dumps(row.get("frequencies", [])), json.dumps(row.get("seasons", [])),
            row.get("start_date"), row.get("end_date"), row.get("location_name"),
            row.get("latitude"), row.get("longitude"), row.get("output_file"),
            row.get("remote_addr"), row.get("status", "success"), row.get("error"),
        ),
    )
    conn.commit()


def next_download_id() -> str:
    today = now_eat().strftime("%Y%m%d")
    prefix = f"CDE-{today}-"
    row = query_one(
        "SELECT download_id FROM export_logs WHERE download_id LIKE ? ORDER BY download_id DESC LIMIT 1",
        (prefix + "%",),
    )
    next_number = 1
    if row and row["download_id"]:
        try:
            next_number = int(str(row["download_id"]).split("-")[-1]) + 1
        except Exception:
            next_number = 1
    return f"{prefix}{next_number:06d}"


def public_base_url() -> str:
    configured = (os.environ.get("CDE_PUBLIC_BASE_URL") or os.environ.get("CDE_BASE_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    root = request.url_root.rstrip("/")
    if root in {"http://127.0.0.1", "http://localhost"}:
        return root + ":5000"
    return root


def make_verification_url(download_id: str) -> str:
    return f"{public_base_url()}{url_for('verify_download', download_id=download_id)}"


def local_base_url() -> str:
    configured = (os.environ.get("CDE_LOCAL_BASE_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    root = request.url_root.rstrip("/")
    if root in {"http://127.0.0.1", "http://localhost"}:
        return root + ":5000"
    return root


def make_data_url(filename: str) -> str:
    # Data URLs intentionally use the local PC/server address that opened the system.
    return f"{local_base_url()}{url_for('download_file', filename=filename)}"


def make_receipt_url(download_id: str) -> str:
    return f"{local_base_url()}{url_for('download_receipt', download_id=download_id)}"


def cde_file_timestamp(value: str | None = None) -> str:
    """Return the DATE_TIME segment used in official CDE file names."""
    if value:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M EAT", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(str(value).replace("Z", "").strip(), fmt)
                break
            except Exception:
                continue
        if parsed is not None:
            return parsed.strftime("%Y%m%d_%H%M%S")
    return now_eat().strftime("%Y%m%d_%H%M%S")


def make_cde_excel_filename(download_id: str, location_name: str, source_key: str, timestamp: str | None = None) -> str:
    """Official exported Excel name.

    The visible file name must start with CDE_ and the current DATE_TIME.
    Do not append the internal CDE download_id here, because that creates a
    second CDE segment in the file name. The download_id remains stored in the
    export log and verification URL.
    """
    stamp = timestamp or cde_file_timestamp()
    safe_loc = secure_filename(location_name)[:45] or "location"
    safe_source = secure_filename(source_key) or "dataset"
    return f"CDE_{stamp}_{safe_loc}_{safe_source}.xlsx"


def make_cde_receipt_filename(download_id: str, row_created_at: str | None = None) -> str:
    """Official Data Delivery Report name.

    The PDF file name starts with CDE_ and DATE_TIME only; the internal
    download_id is intentionally not included in the file name.
    """
    stamp = cde_file_timestamp(row_created_at)
    return f"CDE_{stamp}_Data_Delivery_Report.pdf"


def build_workbook_qr_context(download_id: str, filename: str, customer: Any | None = None) -> Dict[str, Any]:
    downloaded_at = now_eat().strftime("%Y-%m-%d %H:%M EAT")
    staff_name = g.user["full_name"] if g.get("user") else ""
    customer_name = customer_display_name(customer) if customer else ""
    customer_payload = {
        "customer_id": row_value(customer, "id", "") if customer else "",
        "customer_name": customer_name,
        "customer_organization": row_value(customer, "organization", "") if customer else "",
        "customer_phone": row_value(customer, "phone", "") if customer else "",
        "customer_email": row_value(customer, "email", "") if customer else "",
        "customer_address": customer_address_text(customer) if customer else "",
        "cost_recovery_fee": row_value(customer, "cost_recovery_fee", "") if customer else "",
        "customer_remarks": row_value(customer, "remarks", "") if customer else "",
    }
    return {
        "institution": "Climate Data Extractor",
        "system": "Climate Data Extractor",
        "download_id": download_id,
        # Data Delivery Report and QR: Issued By and Served By must both be the logged-in user.
        "served_by": staff_name,
        "downloaded_by": staff_name,
        "issued_by": staff_name,
        "downloaded_at": downloaded_at,
        "file_name": filename,
        "data_url": make_data_url(filename),
        "verification_url": make_verification_url(download_id),
        "receipt_url": make_receipt_url(download_id),
        **customer_payload,
    }

@app.route("/login", methods=["GET", "POST"])
def login():
    if g.get("user") and request.method == "GET":
        return redirect(url_for("home"))
    if request.method == "POST":
        validate_csrf()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = query_one("SELECT * FROM users WHERE lower(email) = lower(?)", (email,))
        otp_ok = False
        password_ok = False
        if user and user["status"] == 1:
            password_ok = check_password_hash(user["password_hash"], password)
            otp_value = (user["otp"] or "").strip() if "otp" in user.keys() else ""
            expires = user["otp_expires_at"] if "otp_expires_at" in user.keys() else None
            if otp_value and secrets.compare_digest(otp_value, password.strip()):
                otp_ok = not expires or expires > datetime.now().isoformat(timespec="seconds")
        if not user or user["status"] != 1 or not (password_ok or otp_ok):
            return render_template("login.html", email=email, inline_error="Invalid email, password or OTP."), 401
        session.clear()
        session.permanent = True
        session["user_id"] = int(user["id"])
        session["csrf_token"] = secrets.token_urlsafe(32)
        now = datetime.now().isoformat(timespec="seconds")
        get_db().execute("UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?", (now, now, user["id"]))
        get_db().commit()
        if otp_ok or user["force_password_change"]:
            flash("Please create your own password before continuing.", "warning")
            return redirect(url_for("change_password"))
        next_url = request.args.get("next") or url_for("home")
        return redirect(next_url)
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    generated_otp = None
    email_value = ""
    if request.method == "POST":
        validate_csrf()
        email_value = (request.form.get("email") or "").strip().lower()
        user = query_one("SELECT * FROM users WHERE lower(email) = lower(?) AND status = 1", (email_value,))
        if user:
            generated_otp = generate_otp()
            now = datetime.now().isoformat(timespec="seconds")
            expires = (datetime.now() + timedelta(hours=24)).isoformat(timespec="seconds")
            get_db().execute(
                """
                UPDATE users
                SET otp = ?, otp_generated_at = ?, otp_expires_at = ?, force_password_change = 1, updated_at = ?
                WHERE id = ?
                """,
                (generated_otp, now, expires, now, user["id"]),
            )
            get_db().commit()
            flash("A new OTP has been generated and saved. Please ask the system administrator for your OTP.", "success")
        else:
            flash("If the email exists, a new OTP will be generated.", "success")
    return render_template("forgot_password.html", generated_otp=None, email_value=email_value)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    conn = get_db()
    candidates = query_all(
        """
        SELECT t.*, u.email, u.full_name
        FROM password_reset_tokens t
        JOIN users u ON u.id = t.user_id
        WHERE t.used_at IS NULL AND t.expires_at > ? AND u.status = 1
        ORDER BY t.created_at DESC
        """,
        (datetime.now().isoformat(timespec="seconds"),),
    )
    matched = None
    for row in candidates:
        if check_password_hash(row["token_hash"], token):
            matched = row
            break
    if not matched:
        flash("Reset link is invalid or expired.", "error")
        return redirect(url_for("forgot_password"))
    if request.method == "POST":
        validate_csrf()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute("UPDATE users SET password_hash = ?, force_password_change = 0, otp = NULL, otp_generated_at = NULL, otp_expires_at = NULL, updated_at = ? WHERE id = ?", (generate_password_hash(password), now, matched["user_id"]))
            conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE id = ?", (now, matched["id"]))
            conn.commit()
            flash("Password changed successfully. Please login.", "success")
            return redirect(url_for("login"))
    return render_template("reset_password.html", token=token, reset_user=matched)


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        validate_csrf()
        current = request.form.get("current_password") or ""
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        otp_value = (g.user["otp"] or "").strip() if "otp" in g.user.keys() else ""
        expires = g.user["otp_expires_at"] if "otp_expires_at" in g.user.keys() else None
        otp_ok = bool(otp_value and secrets.compare_digest(otp_value, current.strip()) and (not expires or expires > datetime.now().isoformat(timespec="seconds")))
        if not (check_password_hash(g.user["password_hash"], current) or otp_ok):
            flash("Current password or OTP is incorrect.", "error")
        elif len(password) < 8:
            flash("New password must contain at least 8 characters.", "error")
        elif password != confirm:
            flash("New passwords do not match.", "error")
        else:
            now = datetime.now().isoformat(timespec="seconds")
            get_db().execute("UPDATE users SET password_hash = ?, force_password_change = 0, otp = NULL, otp_generated_at = NULL, otp_expires_at = NULL, updated_at = ? WHERE id = ?", (generate_password_hash(password), now, g.user["id"]))
            get_db().commit()
            flash("Password changed successfully.", "success")
            return redirect(url_for("home"))
    return render_template("change_password.html")


def catalog_form_sources(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Return form options without scanning or opening any data store."""
    return {
        key: {
            "label": cfg.get("label", key),
            "description": cfg.get("description", ""),
            "supported_frequencies": list(cfg.get("supported_frequencies", [])),
            "variables": dict(cfg.get("variables", {})),
            "available": True,
            "available_by_frequency": {},
        }
        for key, cfg in catalog.items()
    }


def source_summary() -> Dict[str, Any]:
    """Return the configured catalogue without scanning or opening data stores."""
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    sources = catalog_form_sources(catalog)
    total_sources = len(sources)
    total_variables = sum(len(src.get("variables", {})) for src in sources.values())
    configured_files = sum(len(src.get("supported_frequencies", [])) for src in sources.values())
    return {
        "sources": sources,
        "total_sources": total_sources,
        "total_variables": total_variables,
        "available_files": configured_files,
    }


@app.route("/")
@login_required
def home():
    ensure_dirs()
    summary = source_summary()
    return render_template("home.html", **summary)


@app.route("/available-variables")
@login_required
def available_variables():
    ensure_dirs()
    summary = source_summary()
    return render_template("available_variables.html", **summary)


@app.route("/datasets-documentation")
@login_required
def datasets_documentation():
    ensure_dirs()
    summary = source_summary()
    docs = []
    for key, source in summary["sources"].items():
        doc = dict(DATASET_DOCUMENTATION.get(key, {}))
        doc["key"] = key
        doc["label"] = source.get("label", key)
        doc["description"] = source.get("description", "")
        doc["supported_frequencies"] = source.get("supported_frequencies", [])
        doc["file_patterns"] = source.get("file_patterns", {})
        doc["variables"] = source.get("variables", {})
        doc["time_coord"] = source.get("time_coord", "time")
        doc["lat_coord"] = source.get("lat_coord", "latitude")
        doc["lon_coord"] = source.get("lon_coord", "longitude")
        doc["available_by_frequency"] = source.get("available_by_frequency", {})
        docs.append(doc)
    return render_template("datasets_documentation.html", docs=docs, **summary)


@app.route("/data-extractor")
@login_required
def extractor():
    """Data-extraction workspace with saved requests and report preview."""
    ensure_dirs()
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    sources = catalog_form_sources(catalog)
    if request.args.get("reset") == "1":
        session.pop("last_extract_form", None)
    form_data = dict(session.get("last_extract_form") or {}) if request.args.get("preserve") == "1" else {}
    return render_template(
        "extractor.html",
        sources=sources,
        field_errors={},
        form_data=form_data,
        selected_variables=form_data.get("variables", []),
        selected_frequencies=form_data.get("frequencies", []),
        selected_seasons=form_data.get("seasons", []),
    )


@app.route("/api/catalog")
@login_required
def api_catalog():
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    sources = list_available_sources(catalog, DEFAULT_DATA_DIR)
    return jsonify(sources)


@app.route("/api/geocode")
@login_required
def api_geocode():
    q = (request.args.get("q") or "").strip()
    if len(q) < 3:
        return jsonify({"ok": False, "error": "Type a more specific location name."}), 400
    params = urllib.parse.urlencode({"q": q, "format": "jsonv2", "limit": 5, "addressdetails": 1})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "CDE-Satellite-Reanalysis-Platform/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Coordinates could not be fetched automatically. Type a specific name like 'Veyula Dodoma Tanzania' or enter latitude/longitude manually.",
            "detail": str(exc),
        }), 502
    results = []
    for item in data:
        try:
            results.append({"name": item.get("display_name", q), "latitude": float(item.get("lat")), "longitude": float(item.get("lon"))})
        except Exception:
            continue
    if not results:
        return jsonify({
            "ok": False,
            "error": "No coordinates found. Try adding ward, district, region and country, for example: 'Veyula Dodoma Tanzania' or 'Iringa Tanzania'.",
        }), 404
    return jsonify({"ok": True, "results": results})




def parse_requested_locations(form_data, fallback_name: str, fallback_lat: float, fallback_lon: float) -> list[dict[str, Any]]:
    """Parse optional multi-location request from the Data Extractor form."""
    locations: list[dict[str, Any]] = []
    raw = (form_data.get("locations_json") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("location_name") or "").strip()
                    lat = float(item.get("latitude"))
                    lon = float(item.get("longitude"))
                    if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                        locations.append({"name": name, "latitude": lat, "longitude": lon})
        except Exception:
            locations = []
    if not locations:
        locations.append({"name": fallback_name or "Selected Location", "latitude": float(fallback_lat), "longitude": float(fallback_lon)})
    # Keep order and allow repeated stations/locations intentionally.
    return locations


def parse_multi_extraction_requests(raw: str) -> list[dict[str, Any]]:
    """Parse saved Data Extractor request blocks from the browser."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        variables = [str(v).strip() for v in item.get("variables", []) if str(v).strip()]
        frequencies = [str(v).strip() for v in item.get("frequencies", []) if str(v).strip()]
        seasons = [str(v).strip() for v in item.get("seasons", []) if str(v).strip()]
        custom_months = str(item.get("custom_months") or "").strip()
        if custom_months and ("CUSTOM:" + custom_months) not in seasons:
            seasons.append("CUSTOM:" + custom_months)
        start_date = str(item.get("start_date") or "").strip()
        end_date = str(item.get("end_date") or "").strip()
        locations: list[dict[str, Any]] = []
        for loc in item.get("locations", []) or []:
            if not isinstance(loc, dict):
                continue
            name = str(loc.get("name") or loc.get("location_name") or "").strip()
            try:
                lat = float(loc.get("latitude"))
                lon = float(loc.get("longitude"))
            except (TypeError, ValueError):
                continue
            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                locations.append({"name": name, "latitude": lat, "longitude": lon})
        if source and variables and frequencies and start_date and end_date and locations:
            cleaned.append({
                "source": source,
                "variables": variables,
                "frequencies": frequencies,
                "seasons": seasons,
                "custom_months": custom_months,
                "start_date": start_date,
                "end_date": end_date,
                "locations": locations,
            })
    return cleaned



def parse_product_request_bundle(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out = []
    for item in payload:
        if isinstance(item, dict):
            out.append({str(k): v for k, v in item.items()})
    return out

@app.route("/extract", methods=["POST"])
@login_required
@single_product_request
def extract():
    validate_csrf()
    ensure_dirs()
    clean_old_exports()

    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    sources = catalog_form_sources(catalog)
    form_data = request.form
    field_errors: Dict[str, str] = {}

    customer_id = ""
    customer = None

    multi_requests = parse_multi_extraction_requests(request.form.get("requests_json", ""))
    if multi_requests and not field_errors:
        try:
            timestamp = now_eat().strftime("%Y%m%d_%H%M%S")
            download_id = next_download_id()
            excel_filename = f"CDE_{timestamp}_MULTI_REQUESTS_{download_id}.xlsx"
            excel_path = DEFAULT_EXPORT_DIR / excel_filename
            first_req = multi_requests[0]
            first_locations = first_req.get("locations") or []
            first_location_name = first_locations[0].get("name", "Multiple Locations") if first_locations else "Multiple Locations"

            for idx, req in enumerate(multi_requests, start=1):
                req_source_key = req["source"]
                req_source_cfg = catalog.get(req_source_key)
                if req_source_cfg is None:
                    raise ValueError(f"Saved request {idx} uses an unavailable dataset.")
                bad_vars = [v for v in req["variables"] if v not in req_source_cfg.get("variables", {})]
                if bad_vars:
                    raise ValueError(f"Saved request {idx} contains unavailable weather elements.")
                supported = set(req_source_cfg.get("supported_frequencies", []))
                invalid_freq = sorted(set(req["frequencies"]) - supported)
                if invalid_freq:
                    raise ValueError(f"Saved request {idx} uses unsupported temporal resolution: {', '.join(invalid_freq)}")
                if req["start_date"] > req["end_date"]:
                    raise ValueError(f"Saved request {idx} has start date after end date.")
                stations = [
                    StationPoint(
                        station_id=f"REQ{idx}_POINT_{j}",
                        station_name=loc["name"],
                        latitude=float(loc["latitude"]),
                        longitude=float(loc["longitude"]),
                    )
                    for j, loc in enumerate(req["locations"], start=1)
                ]
                req_download_id = f"{download_id}-{idx:02d}"
                qr_context = build_workbook_qr_context(req_download_id, excel_filename, customer=customer)
                write_excel_output(
                    output_path=excel_path,
                    source_key=req_source_key,
                    source_cfg=req_source_cfg,
                    stations=stations,
                    variables=req["variables"],
                    start_date=req["start_date"],
                    end_date=req["end_date"],
                    frequencies=req["frequencies"],
                    seasons=req.get("seasons") or [],
                    data_dir=DEFAULT_DATA_DIR,
                    download_context=qr_context,
                    append=idx > 1,
                )

            data_url = make_data_url(excel_filename)
            verification_url = make_verification_url(download_id)
            receipt_url = make_receipt_url(download_id)
            request_lines = []
            station_names = []
            element_names = []
            data_types = []
            for ridx, req in enumerate(multi_requests, start=1):
                cfg = catalog.get(req.get("source"), {})
                src_label = cfg.get("label", req.get("source", ""))
                vars_labels = [cfg.get("variables", {}).get(v, {}).get("label", v) for v in req.get("variables", [])]
                loc_names = [loc.get("name", "") for loc in req.get("locations", [])]
                station_names.extend([x for x in loc_names if x])
                element_names.extend(vars_labels)
                data_types.extend(req.get("frequencies", []))
                request_lines.append(f"Request {ridx}: {src_label}; {', '.join(vars_labels)}; {', '.join(req.get('frequencies', []))}; {', '.join(loc_names)}; {req.get('start_date','')} to {req.get('end_date','')}")
            qr_payload = {
                "institution": "Climate Data Extractor",
                "document_type": "Data Delivery Report",
                "download_id": download_id,
                "request_no": download_id,
                "customer_name": customer_display_name(customer),
                "customer_organization": row_value(customer, "organization", ""),
                "customer_phone": row_value(customer, "phone", ""),
                "customer_email": row_value(customer, "email", ""),
                "customer_address": customer_address_text(customer),
                "station_name": "; ".join(station_names),
                "source": "Multiple datasets/requests",
                "element": "; ".join(dict.fromkeys(element_names)),
                "data_type": ", ".join(dict.fromkeys(data_types)),
                "start_date": first_req.get("start_date", ""),
                "end_date": first_req.get("end_date", ""),
                "period": f"{first_req.get('start_date','')} to {first_req.get('end_date','')}",
                "served_by": g.user["full_name"] if g.get("user") else "",
                "issued_by": g.user["full_name"] if g.get("user") else "",
                "file_name": excel_filename,
                "data_url": data_url,
                "verification_url": verification_url,
                "receipt_url": receipt_url,
                "request_details": " | ".join(request_lines),
            }

            log_export({
                "user_id": g.user["id"],
                "customer_id": row_value(customer, "id", None),
                "customer_name": customer_display_name(customer),
                "customer_organization": row_value(customer, "organization", ""),
                "customer_phone": row_value(customer, "phone", ""),
                "customer_email": row_value(customer, "email", ""),
                "customer_address": customer_address_text(customer),
                "issued_by": g.user["full_name"] if g.get("user") else "",
                "source": "Multiple",
                "variables": [", ".join(req.get("variables", [])) for req in multi_requests],
                "frequencies": [", ".join(req.get("frequencies", [])) for req in multi_requests],
                "seasons": [", ".join(req.get("seasons", [])) for req in multi_requests],
                "start_date": first_req.get("start_date", ""),
                "end_date": first_req.get("end_date", ""),
                "location_name": first_location_name if len(multi_requests) == 1 else f"{len(multi_requests)} Requests",
                "latitude": first_locations[0].get("latitude") if first_locations else "",
                "longitude": first_locations[0].get("longitude") if first_locations else "",
                "output_file": excel_filename,
                "remote_addr": request.remote_addr,
                "status": "success",
                "download_id": download_id,
                "verification_url": verification_url,
                "qr_payload": json.dumps(qr_payload, ensure_ascii=False),
            })
            session["last_extract_form"] = dict(request.form)
            return render_template(
                "result.html", success=True, filename=excel_filename,
                download_url=url_for("download_file", filename=excel_filename),
                download_id=download_id, verification_url=verification_url, receipt_url=receipt_url, data_url=data_url,
                source="Multiple datasets/requests",
                variables=[f"{len(multi_requests)} saved request(s)"],
                station_count=sum(len(req.get("locations", [])) for req in multi_requests),
                frequencies=[], seasons=[], start_date=first_req.get("start_date", ""), end_date=first_req.get("end_date", ""),
                location_name=f"{len(multi_requests)} saved request(s)",
                customer_name=customer_display_name(customer),
            )
        except Exception as exc:
            traceback.print_exc()
            field_errors["generationField"] = str(exc)
            return render_template(
                "extractor.html",
                sources=sources,
                        field_errors=field_errors,
                form_data=form_data,
                selected_variables=request.form.getlist("variables"),
                selected_frequencies=request.form.getlist("frequencies"),
                selected_seasons=request.form.getlist("seasons"),
            ), 400

    source_key = request.form.get("source", "").strip()
    source_cfg = catalog.get(source_key)
    if not source_key or source_key not in catalog:
        field_errors["sourceField"] = "Please select a valid data source."

    variables = [v.strip() for v in request.form.getlist("variables") if v.strip()]
    if not variables:
        field_errors["variablesField"] = "Please select at least one variable."
    elif source_cfg:
        bad_vars = [v for v in variables if v not in source_cfg.get("variables", {})]
        if bad_vars:
            field_errors["variablesField"] = "One or more selected variables are not available for this data source."

    start_date = request.form.get("start_date", "").strip()
    end_date = request.form.get("end_date", "").strip()
    if not start_date:
        field_errors["startDateField"] = "Start date is required."
    if not end_date:
        field_errors["endDateField"] = "End date is required."
    if start_date and end_date and start_date > end_date:
        field_errors["startDateField"] = "Start date cannot be after end date."
        field_errors["endDateField"] = "End date must be after start date."

    frequencies = [f.strip() for f in request.form.getlist("frequencies") if f.strip()]
    if not frequencies:
        field_errors["frequencyField"] = "Please select at least one temporal resolution."
    elif source_cfg:
        supported = set(source_cfg.get("supported_frequencies", []))
        invalid = sorted(set(frequencies) - supported)
        if invalid:
            field_errors["frequencyField"] = f"{source_cfg.get('label', source_key)} does not support: {', '.join(invalid)}"

    seasons = [s.strip() for s in request.form.getlist("seasons") if s.strip()]
    custom_months = request.form.get("custom_months", "").strip()
    if custom_months:
        custom_parts = [p for p in re.split(r"[\s,;|/]+", custom_months) if p]
        try:
            custom_month_values = []
            for part in custom_parts:
                if not part.isdigit():
                    raise ValueError
                month = int(part)
                if month < 1 or month > 12:
                    raise ValueError
                if month not in custom_month_values:
                    custom_month_values.append(month)
            if not custom_month_values:
                raise ValueError
            custom_months = ",".join(str(m) for m in custom_month_values)
            seasons.append("CUSTOM:" + custom_months)
        except ValueError:
            field_errors["seasonField"] = "Custom season must use month numbers from 1 to 12, for example 1,2,3."
    if "seasonal" in frequencies and not seasons:
        field_errors["seasonField"] = "Select at least one seasonal option or enter custom months."

    location_name = request.form.get("location_name", "").strip()
    # Station ID is intentionally not collected on the Data Extractor page.
    # Point-based Zarr exports use an internal ID only.
    station_id = "POINT_1"
    if not location_name:
        field_errors["locationNameField"] = "Location or station name is required."

    lat_raw = request.form.get("latitude", "").strip()
    lon_raw = request.form.get("longitude", "").strip()
    latitude = None
    longitude = None
    if not lat_raw:
        field_errors["latitudeField"] = "Latitude is required. Search a location or type it manually."
    else:
        try:
            latitude = float(lat_raw)
            if latitude < -90 or latitude > 90:
                field_errors["latitudeField"] = "Latitude must be between -90 and 90."
        except ValueError:
            field_errors["latitudeField"] = "Latitude must be a valid number."
    if not lon_raw:
        field_errors["longitudeField"] = "Longitude is required. Search a location or type it manually."
    else:
        try:
            longitude = float(lon_raw)
            if longitude < -180 or longitude > 180:
                field_errors["longitudeField"] = "Longitude must be between -180 and 180."
        except ValueError:
            field_errors["longitudeField"] = "Longitude must be a valid number."

    if field_errors:
        return render_template(
            "extractor.html",
            sources=sources,
                field_errors=field_errors,
            form_data=form_data,
            selected_variables=variables,
            selected_frequencies=frequencies,
            selected_seasons=request.form.getlist("seasons"),
        ), 400

    assert source_cfg is not None
    assert latitude is not None and longitude is not None
    try:
        location_name = location_name or "Custom Location"
        requested_locations = parse_requested_locations(request.form, location_name, latitude, longitude)
        stations = [
            StationPoint(
                station_id=f"POINT_{i}",
                station_name=item["name"],
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
            )
            for i, item in enumerate(requested_locations, start=1)
        ]
        export_location_name = location_name if len(stations) == 1 else f"{len(stations)}_Locations"
        timestamp = now_eat().strftime("%Y%m%d_%H%M%S")
        download_id = next_download_id()
        filename = make_cde_excel_filename(download_id, export_location_name, source_key, timestamp)
        output_path = DEFAULT_EXPORT_DIR / filename
        qr_context = build_workbook_qr_context(download_id, filename, customer=customer)
        write_excel_output(
            output_path=output_path,
            source_key=source_key,
            source_cfg=source_cfg,
            stations=stations,
            variables=variables,
            start_date=start_date,
            end_date=end_date,
            frequencies=frequencies,
            seasons=seasons,
            data_dir=DEFAULT_DATA_DIR,
            download_context=qr_context,
        )
        # Parquet generation disabled for Data Extractor downloads by user request.
        session["last_extract_form"] = {
            "source": source_key,
            "variables": variables,
            "frequencies": frequencies,
            "seasons": request.form.getlist("seasons"),
            "custom_months": custom_months,
            "start_date": start_date,
            "end_date": end_date,
            "location_name": location_name,
            "latitude": lat_raw,
            "longitude": lon_raw,
            "locations_json": request.form.get("locations_json", ""),
        }
        variable_labels = [source_cfg["variables"][v].get("label", v) for v in variables]
        variable_units = [source_cfg["variables"][v].get("unit", "") for v in variables]
        qr_log_payload = {
            **qr_context,
            "customer_name": customer_display_name(customer),
            "customer_organization": row_value(customer, "organization", ""),
            "customer_phone": row_value(customer, "phone", ""),
            "customer_email": row_value(customer, "email", ""),
            "customer_address": customer_address_text(customer),
            "cost_recovery_fee": row_value(customer, "cost_recovery_fee", ""),
            "customer_remarks": row_value(customer, "remarks", ""),
            "issued_by": g.user["full_name"] if g.get("user") else "",
            "station_name": ", ".join([st.station_name for st in stations]),
            "latitude": stations[0].latitude if stations else latitude,
            "longitude": stations[0].longitude if stations else longitude,
            "source": source_cfg.get("label", source_key),
            "element": variable_labels[0] if len(variable_labels) == 1 else ", ".join(variable_labels),
            "elements": variable_labels,
            "data_type": data_type_label(frequencies[0], seasons[0] if frequencies[0] == "seasonal" and seasons else None) if len(frequencies) == 1 else ", ".join(data_type_label(freq) for freq in frequencies),
            "data_types": frequencies,
            "seasons": seasons,
            "start_date": start_date,
            "end_date": end_date,
            "units": variable_units[0] if len(set(variable_units)) == 1 else ", ".join([u for u in variable_units if u]),
        }
        log_export({
            "user_id": g.user["id"],
            "customer_id": row_value(customer, "id", None),
            "customer_name": customer_display_name(customer),
            "customer_organization": row_value(customer, "organization", ""),
            "customer_phone": row_value(customer, "phone", ""),
            "customer_email": row_value(customer, "email", ""),
            "customer_address": customer_address_text(customer),
            "cost_recovery_fee": row_value(customer, "cost_recovery_fee", ""),
            "customer_remarks": row_value(customer, "remarks", ""),
            "issued_by": g.user["full_name"] if g.get("user") else "",
            "source": source_key, "variables": variables,
            "frequencies": frequencies, "seasons": seasons, "start_date": start_date,
            "end_date": end_date, "location_name": ", ".join([st.station_name for st in stations]), "latitude": stations[0].latitude if stations else latitude,
            "longitude": stations[0].longitude if stations else longitude, "output_file": filename, "remote_addr": request.remote_addr,
            "status": "success", "download_id": download_id,
            "verification_url": qr_context.get("verification_url"),
            "qr_payload": json.dumps(qr_log_payload, ensure_ascii=False),
        })
        return render_template(
            "result.html", success=True, filename=filename,
            download_url=url_for("download_file", filename=filename),
            download_id=download_id, verification_url=qr_context.get("verification_url"),
            receipt_url=qr_context.get("receipt_url"), data_url=qr_context.get("data_url"),
            source=source_cfg.get("label", source_key),
            variables=variable_labels,
            station_count=len(stations), frequencies=frequencies, seasons=seasons,
            start_date=start_date, end_date=end_date, location_name=", ".join([st.station_name for st in stations]),
            customer_name=customer_display_name(customer),
        )
    except Exception as exc:
        error_text = str(exc)
        traceback.print_exc()
        log_export({
            "user_id": g.user["id"],
            "customer_id": row_value(customer, "id", None) if customer else None,
            "customer_name": customer_display_name(customer) if customer else "",
            "customer_organization": row_value(customer, "organization", "") if customer else "",
            "customer_phone": row_value(customer, "phone", "") if customer else "",
            "customer_email": row_value(customer, "email", "") if customer else "",
            "customer_address": customer_address_text(customer) if customer else "",
            "cost_recovery_fee": row_value(customer, "cost_recovery_fee", "") if customer else "",
            "customer_remarks": row_value(customer, "remarks", "") if customer else "",
            "issued_by": g.user["full_name"] if g.get("user") else "",
            "status": "error", "error": error_text,
            "remote_addr": request.remote_addr, "source": source_key,
            "variables": variables, "frequencies": frequencies, "seasons": seasons,
            "start_date": start_date, "end_date": end_date, "location_name": location_name,
            "latitude": latitude, "longitude": longitude,
        })
        field_errors = {"generationField": error_text}
        return render_template(
            "extractor.html",
            sources=sources,
                field_errors=field_errors,
            form_data=form_data,
            selected_variables=variables,
            selected_frequencies=frequencies,
            selected_seasons=request.form.getlist("seasons"),
        ), 400






def _relative_export_path(path: Path) -> str:
    """Return a safe path relative to DEFAULT_EXPORT_DIR for product downloads."""
    path = Path(path).resolve()
    base = DEFAULT_EXPORT_DIR.resolve()
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return path.name


@app.route("/product-download/<path:filename>")
@login_required
def product_download(filename: str):
    base = DEFAULT_EXPORT_DIR.resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)) or not target.exists() or not target.is_file():
        abort(404)
    if target.suffix.lower() not in {".xlsx", ".png", ".pdf"}:
        abort(404)
    mimetype = "application/octet-stream"
    if target.suffix.lower() == ".png":
        mimetype = "image/png"
    elif target.suffix.lower() == ".xlsx":
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif target.suffix.lower() == ".parquet":
        mimetype = "application/octet-stream"
    elif target.suffix.lower() == ".pdf":
        mimetype = "application/pdf"
    as_attachment = not (target.suffix.lower() in {".png", ".pdf"} and request.args.get("inline") == "1")
    return send_file(target, as_attachment=as_attachment, download_name=target.name, mimetype=mimetype, max_age=0)




def _safe_filename_piece(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_") or "record"


def _relative_parquet_db_path(path: Path) -> str:
    path = Path(path).resolve()
    base = PARQUET_DB_DIR.resolve()
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return path.name


def _resolve_parquet_db_path(relative_name: str) -> Path:
    base = PARQUET_DB_DIR.resolve()
    target = (base / relative_name).resolve()
    if not str(target).startswith(str(base)) or not target.exists() or target.suffix.lower() != ".parquet":
        abort(404)
    return target


def _read_parquet_preview(path: Path, limit: int = 100) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if limit and len(df) > limit:
        return df.head(limit).copy()
    return df.copy()


def _dataframe_for_pdf(path: Path, max_rows: int = 500) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
        df["_note"] = f"Preview only: first {max_rows} records shown in PDF. Full database record contains more rows."
    return df


def build_records_pdf(title: str, df: pd.DataFrame, metadata: Dict[str, Any] | None = None) -> bytes:
    """Convert selected Parquet records/summary into a professional PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.utils import ImageReader
    import textwrap

    buffer = BytesIO()
    page_size = landscape(A4) if len(df.columns) > 6 else A4
    doc = SimpleDocTemplate(buffer, pagesize=page_size, rightMargin=12 * mm, leftMargin=12 * mm, topMargin=12 * mm, bottomMargin=12 * mm)
    styles = getSampleStyleSheet()
    story = []

    header_rows = []
    story.append(Paragraph("<b>TANZANIA METEOROLOGICAL AUTHORITY</b>", styles["Title"]))
    story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    story.append(Paragraph(f"Generated: {now_eat().strftime('%d %B %Y %H:%M EAT')}", styles["Normal"]))
    if metadata:
        meta_text = " &nbsp; | &nbsp; ".join([f"<b>{k}:</b> {v}" for k, v in metadata.items() if v not in [None, ""]])
        if meta_text:
            story.append(Paragraph(meta_text, styles["Normal"]))
    story.append(Spacer(1, 5 * mm))

    if df.empty:
        story.append(Paragraph("No records found in the selected database file.", styles["Normal"]))
    else:
        pdf_df = df.copy()
        # Keep PDF readable by limiting columns and shortening text.
        max_cols = 10
        if len(pdf_df.columns) > max_cols:
            visible_cols = list(pdf_df.columns[:max_cols])
            pdf_df = pdf_df[visible_cols]
        pdf_df = pdf_df.fillna("")
        headers = [Paragraph(f"<b>{str(c)}</b>", styles["Normal"]) for c in pdf_df.columns]
        data = [headers]
        for _, row in pdf_df.iterrows():
            data.append([Paragraph(str(v)[:120], styles["Normal"]) for v in row.tolist()])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E7490")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(table)
        if len(df.columns) > max_cols:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(f"Only the first {max_cols} columns are shown for readability.", styles["Italic"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("This PDF was generated from records saved in the CDE database.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def save_records_pdf_from_parquet(parquet_path: Path, out_dir: Path, title: str, metadata: Dict[str, Any] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _dataframe_for_pdf(parquet_path)
    pdf_bytes = build_records_pdf(title, df, metadata or {})
    pdf_path = out_dir / f"{Path(parquet_path).stem}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path




def save_workbook_sheets_to_parquet(workbook_path: Path, download_id: str, metadata: Dict[str, Any]) -> list[Path]:
    """Persist generated data-extractor workbook sheets to the Parquet database.

    The Excel remains the user-facing download, while the saved database copy is Parquet.
    This keeps source Zarr stores untouched and gives CDE a queryable file-based database.
    """
    saved: list[Path] = []
    base = PARQUET_DB_DIR / "data_extractions" / _safe_filename_piece(download_id)
    base.mkdir(parents=True, exist_ok=True)
    try:
        sheets = pd.read_excel(workbook_path, sheet_name=None, dtype=str)
    except Exception:
        return saved
    manifest_rows = []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.insert(0, "sheet_name", sheet_name)
        for key, value in metadata.items():
            df[key] = value
        out = base / f"{_safe_filename_piece(sheet_name)}.parquet"
        df.to_parquet(out, index=False)
        saved.append(out)
        manifest_rows.append({"download_id": download_id, "sheet_name": sheet_name, "rows": len(df), "parquet_file": str(out.name), **metadata})
    if manifest_rows:
        manifest = base / "manifest.parquet"
        pd.DataFrame(manifest_rows).to_parquet(manifest, index=False)
        saved.append(manifest)
    return saved

def _bounded_product_years(start_year: str, end_year: str, resolution: str, mode: str) -> tuple[int, int]:
    """Validate the requested year range without imposing a processing cap."""
    try:
        start = int(str(start_year).strip())
        end = int(str(end_year).strip())
    except Exception as exc:
        raise ValueError("Start year and end year must be valid years.") from exc
    if start > end:
        raise ValueError("Start year cannot be after end year.")
    return start, end


ANALYTICAL_PRODUCT_TYPES = {
    "data_extraction", "statistical_summary", "climatology_profile",
    "variability_analysis", "trend_variability", "extremes_analysis", "comprehensive_analysis",
    "long_term_annual_trend_analysis",
}


INDEX_OPTIONS_BY_DATASET = {
    "chirps_rainfall": RAINFALL_INDICES,
    "era5_total_precipitation": RAINFALL_INDICES,
    "era5_temperature": TEMPERATURE_INDICES,
    "era5_temperature_stats": [
        (key, label) for key, label in TEMPERATURE_INDICES
        if key in {
            "mean_temperature", "maximum_temperature", "minimum_temperature",
            "temperature_anomaly", "hot_days", "hot_nights",
            "cold_days", "cold_nights", "dtr"
        }
    ],
    "era5_relative_humidity": [(key, label) for key, label in OTHER_INDICES if key == "relative_humidity_index"],
    "era5_soil_water": [(key, label) for key, label in OTHER_INDICES if key in {"soil_moisture_index", "soil_moisture_anomaly"}],
    "era5_wind": [(key, label) for key, label in OTHER_INDICES if key == "wind_speed_index"],
}

PRODUCT_DESCRIPTIONS = {
    "data_extraction": "Extract the selected period and location with quality-control metadata and reusable tabular outputs.",
    "time_series": "Review how the selected weather element changes through time.",
    "bar": "Compare aggregated values across the selected temporal groups.",
    "spatial_map": "Map the selected weather element across Tanzania for one selected period.",
    "spatial_climatology": "Map the long-term spatial climatology for independently selected months and years.",
    "spatial_monthly_climatology": "Generate January–December spatial climatology maps in one figure.",
    "spatial_annual_series": "Generate one annual spatial map for each selected year.",
    "spatial_seasonal_climatology": "Generate seasonal spatial climatology maps in one figure.",
    "heatmap": "Examine month-to-year patterns and temporal concentration.",
    "annual_trend": "Assess annual behaviour with a fitted long-term trend.",
    "anomaly": "Compare annual conditions against the selected climatological baseline.",
    "wind_rose": "Summarise wind-direction frequency and wind-speed classes.",
    "histogram": "Review the frequency distribution of the selected observations.",
    "area": "Display cumulative precipitation through the selected period.",
    "multi_line": "Compare two or more weather elements on one time axis.",
    "monthly_climatology": "Show the average monthly total for precipitation or monthly mean for other weather elements from January to December.",
    "temperature_monthly_profile": "Compare mean, minimum and maximum temperature from January to December.",
    "seasonal_profile": "Compare DJF, MAM, JJA and SON climatological conditions.",
    "standardized_anomaly": "Express annual departures from the baseline in standard-deviation units.",
    "spatial_std_map": "Map temporal standard deviation across Tanzania.",
    "spatial_cv_map": "Map coefficient of variation across Tanzania.",
    "mean_std_band": "Show the monthly mean and a shaded ±1 standard-deviation band.",
    "std_error_bars": "Show monthly means with standard-deviation error bars.",
    "standard_deviation": "Compare monthly standard deviation values.",
    "coefficient_variation": "Compare variability relative to the mean.",
    "box": "Summarise monthly distributions using quartiles and whiskers.",
    "extreme_value": "Compare annual maximum values.",
    "scatter": "Examine the relationship between two selected weather elements.",
    "statistical_summary": "Summarise central tendency, variability, percentiles, distribution and data quality.",
    "climatology_profile": "Assess monthly, seasonal and decadal climatological behaviour.",
    "variability_analysis": "Assess standard deviation, coefficient of variation and standardized anomalies.",
    "trend_variability": "Evaluate anomalies, moving averages, linear trend, Mann–Kendall significance and Sen's slope.",
    "extremes_analysis": "Assess observed extremes, percentile thresholds and annual climate indicators.",
    "comprehensive_analysis": "Produce an integrated extraction, climatology, trend, variability and extremes package.",
    "spatial_climatology_analysis": "Produce monthly, annual and seasonal spatial climatology map products.",
    "step_plot": "Display changes as discrete steps through time.",
    "lollipop": "Compare individual time periods using stems and markers.",
    "rolling_mean": "Show observations together with a moving average.",
    "cumulative_total": "Show cumulative precipitation through the selected period.",
    "percentile_band": "Show monthly median and 10th–90th percentile envelopes.",
    "exceedance_curve": "Show how often values are equalled or exceeded.",
    "ecdf": "Show the empirical cumulative distribution of observations.",
    "violin": "Compare monthly distribution shapes and central values.",
    "diurnal_cycle": "Show the average hourly cycle with standard-deviation spread.",
    "rank_plot": "Rank selected values from highest to lowest.",
    "data_quality_analysis": "Assess completeness, valid observations and temporal gaps.",
    "distribution_analysis": "Assess distribution, quantiles, skewness and outliers.",
    "percentile_analysis": "Assess percentile thresholds and exceedance frequencies.",
    "monthly_variability_analysis": "Assess monthly standard deviation, coefficient of variation and percentile spread.",
    "seasonal_comparison_analysis": "Compare climatological conditions among seasons.",
    "decadal_change_analysis": "Compare decadal means and long-term changes.",
    "anomaly_analysis": "Assess absolute, percentage and standardized anomalies.",
    "trend_significance_analysis": "Assess trend magnitude and statistical significance.",
    "long_term_annual_trend_analysis": "Aggregate precipitation to annual totals or other variables to annual means, then fit and report a linear trend equation, R², Mann–Kendall test and Sen's slope.",
    "extreme_frequency_analysis": "Assess frequencies of unusually high and low values.",
}



def _professional_product_heading(params: dict, result: dict) -> tuple[str, str]:
    plot_type = str(params.get("plot_type") or "time_series")
    label = dict(PLOT_TYPES).get(plot_type, plot_type.replace("_", " ").title())
    context = result.get("context") or {}
    element = variable_display_name(params.get("variable"), context, params.get("dataset"))
    location = str(params.get("location_name") or "Selected Location")
    start_year = str(params.get("start_year") or "")
    end_year = str(params.get("end_year") or "")
    resolution = str(params.get("resolution") or "").capitalize()
    period_label = str(context.get("period_label") or result.get("period_label") or "").strip()
    custom_title = " ".join(str(params.get("custom_plot_title") or "").split())[:180]

    if custom_title:
        title = custom_title
    elif plot_type == "data_extraction":
        title = f"{element} Data Extraction for {location}"
    elif plot_type in {"spatial_map", "spatial_std_map", "spatial_cv_map", "spatial_climatology", "spatial_monthly_climatology", "spatial_annual_series", "spatial_seasonal_climatology"}:
        title = f"{label}: {element} over Tanzania"
    elif plot_type == "heatmap":
        title = f"{element} Monthly–Annual Heat Map for {location}"
    else:
        title = f"{label}: {element} for {location}"

    if period_label:
        subtitle = period_label
    else:
        subtitle = f"{resolution} product"
        if start_year and end_year:
            subtitle += f" · {start_year}–{end_year}"
        if params.get("season") and str(params.get("resolution")) == "seasonal":
            subtitle += f" · {params.get('season')} season"
    admin_label = context.get("administrative_level_label")
    if admin_label:
        subtitle += f" · {admin_label}"
    return title, subtitle


@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    """Compatibility redirect into the single climate-analysis service."""
    return redirect(url_for("plots", service="analysis"), code=302)


@app.route("/indices", methods=["GET", "POST"])
@login_required
def indices():
    """Compatibility redirect into the climate-index service."""
    return redirect(url_for("plots", service="index"), code=307 if request.method == "POST" else 302)


@app.route("/plots", methods=["GET", "POST"])
@login_required
@single_product_request
def plots():
    """Single request-based workspace for extraction, visual products, analysis and indices."""
    ensure_dirs()
    sources = product_form_sources()
    allowed_services = {"extraction", "visual", "analysis", "index"}
    requested_service = str(
        request.form.get("workspace_service")
        or request.args.get("service")
        or "extraction"
    ).strip().lower()
    service = requested_service if requested_service in allowed_services else "extraction"

    # A GET or an unrelated POST must never open a Zarr store or generate a
    # product. Only the explicit Generate/Run button is allowed to start heavy
    # processing. This also prevents browser form restoration or extensions
    # from accidentally replaying an expensive request.
    if request.method == "POST" and request.form.get("generate_requested") != "1":
        return redirect(url_for("plots", service=service), code=303)

    visual_products = {
        "time_series", "bar", "area", "multi_line", "monthly_climatology", "seasonal_profile",
        "annual_trend", "anomaly", "standardized_anomaly", "step_plot", "lollipop",
        "rolling_mean", "cumulative_total", "percentile_band", "exceedance_curve", "ecdf",
        "violin", "diurnal_cycle", "rank_plot", "spatial_map", "spatial_std_map",
        "spatial_cv_map", "heatmap", "mean_std_band", "std_error_bars", "standard_deviation",
        "coefficient_variation", "histogram", "box", "extreme_value", "scatter", "wind_rose",
        "temperature_monthly_profile", "spatial_climatology", "spatial_monthly_climatology",
        "spatial_annual_series", "spatial_seasonal_climatology",
    }
    analysis_products = {"statistical_summary", "climatology_profile", "variability_analysis", "trend_variability", "extremes_analysis", "comprehensive_analysis", "spatial_climatology_analysis", "data_quality_analysis", "distribution_analysis", "percentile_analysis", "monthly_variability_analysis", "seasonal_comparison_analysis", "decadal_change_analysis", "anomaly_analysis", "trend_significance_analysis", "long_term_annual_trend_analysis", "extreme_frequency_analysis"}
    requested_product = str(request.args.get("product") or "").strip()
    if requested_product in analysis_products:
        service = "analysis"
    elif requested_product in visual_products:
        service = "visual"
    elif requested_product == "data_extraction":
        service = "extraction"

    # Extraction requests use the customer-aware delivery workspace so every
    # generated dataset has saved-request support, a Data Delivery Report and
    # page-level QR verification. Plot, analysis and index requests remain here.
    if service == "extraction" and request.method == "GET":
        return redirect(url_for("extractor"), code=302)

    default_dataset = "chirps_rainfall"
    defaults = {
        "workspace_service": service,
        "plot_dataset": default_dataset,
        "visual_type": requested_product if requested_product in visual_products else "time_series",
        "analysis_type": requested_product if requested_product in analysis_products else "statistical_summary",
        "index_type": "total_rainfall",
        "index_dataset": "chirps_rainfall",
        "index_season": "ANNUAL",
        "index_resolution": "annual",
        "index_plot_type": "auto",
        "start_date": "1991-01-01",
        "end_date": "2025-12-31" if service == "index" else "2020-12-31",
        "start_hour": "0",
        "end_hour": "23",
        "index_custom_years": "",
        "custom_months": "",
        "start_year": "1991",
        "end_year": "2020" if service != "index" else "2025",
        "map_period_mode": "custom",
        "map_month_selection": "all",
        "map_year_selection": "range",
        "map_year": "2020",
        "map_start_year": "1991",
        "map_end_year": "2020",
        "map_month": "01",
        "map_season": "MAM",
        "map_custom_months": "1,2,3",
        "map_custom_years": "",
        "map_admin_level": "1",
        "map_output_layout": "single",
        "map_panel_basis": "auto",
        "map_render_style": "grid",
        "show_lakes": "1",
        "heatmap_month_mode": "all",
        "heatmap_custom_months": "1,2,3",
        "heatmap_season": "MAM",
        "heatmap_year_mode": "range",
        "heatmap_custom_years": "",
        "baseline_start": "1991",
        "baseline_end": "2020",
        "latitude": "-6.1731",
        "longitude": "35.7417",
        "location_name": "Dodoma",
        "rainy_threshold": "1",
        "heavy_threshold": "50",
        "very_heavy_threshold": "100",
        "heat_threshold": "35",
        "warm_threshold": "30",
        "cold_threshold": "15",
        "include_plot": "0",
        "custom_plot_title": "",
    }
    submitted = request.form.to_dict(flat=True) if request.method == "POST" else {}
    form_data = {**defaults, **submitted}
    form_data["workspace_service"] = service
    if request.method == "POST":
        for key in ("show_lakes", "include_plot"):
            form_data[key] = "1" if request.form.get(key) else "0"

    dataset = str(form_data.get("plot_dataset") or default_dataset)
    valid_dataset_keys = set(INDEX_OPTIONS_BY_DATASET) if service == "index" else set(PRODUCT_DATASETS)
    if dataset not in valid_dataset_keys:
        dataset = default_dataset
        form_data["plot_dataset"] = dataset
    allowed_variables = [item["value"] for item in MULTI_VARIABLE_OPTIONS.get(dataset, [])]
    allowed_resolutions = list(PRODUCT_DATASETS[dataset].get("resolutions", {}).keys())

    selected_variables = request.form.getlist("plot_variables") if request.method == "POST" else allowed_variables[:1]
    selected_variables = [v for v in selected_variables if v in allowed_variables]
    if not selected_variables and allowed_variables:
        selected_variables = allowed_variables[:1]

    selected_resolutions = request.form.getlist("plot_resolutions") if request.method == "POST" else (["monthly"] if "monthly" in allowed_resolutions else allowed_resolutions[:1])
    selected_resolutions = [r for r in selected_resolutions if r in allowed_resolutions]
    if not selected_resolutions and allowed_resolutions:
        selected_resolutions = ["monthly"] if "monthly" in allowed_resolutions else allowed_resolutions[:1]

    selected_seasons = request.form.getlist("plot_seasons") if request.method == "POST" else ["MAM"]
    selected_seasons = [s for s in selected_seasons if s in PRODUCT_SEASONS]
    if not selected_seasons:
        selected_seasons = ["MAM"]

    result = None
    result_items: list[dict[str, Any]] = []
    combined_excel_url: str | None = None
    field_errors: dict[str, str] = {}

    if request.method == "POST":
        validate_csrf()
        try:
            visual_type_for_validation = str(form_data.get("visual_type") or "time_series")
            spatial_visual_products = {"spatial_map", "spatial_std_map", "spatial_cv_map", "spatial_climatology", "spatial_monthly_climatology", "spatial_annual_series", "spatial_seasonal_climatology"}
            is_spatial_map = (service == "visual" and visual_type_for_validation in spatial_visual_products) or (service == "analysis" and str(form_data.get("analysis_type") or "") == "spatial_climatology_analysis")
            required_fields = [] if is_spatial_map else ["location_name", "latitude", "longitude", "start_date", "end_date"]
            for required in required_fields:
                if not str(form_data.get(required, "")).strip():
                    field_errors[required] = "This field is required."
            if field_errors:
                raise ValueError("Complete the location, coordinates and period fields.")

            if is_spatial_map:
                start_date = f"{int(form_data.get('start_year') or 1991)}-01-01"
                end_date = f"{int(form_data.get('end_year') or 2020)}-12-31"
            else:
                period_start = pd.Timestamp(str(form_data.get("start_date")))
                period_end = pd.Timestamp(str(form_data.get("end_date")))
                if period_start > period_end:
                    raise ValueError("The start date cannot be after the end date.")
                start_hour = max(0, min(23, int(form_data.get("start_hour") or 0)))
                end_hour = max(0, min(23, int(form_data.get("end_hour") or 23)))
                if any(value == "hourly" for value in (selected_resolutions if service != "index" else [str(form_data.get("index_resolution") or "annual")])):
                    period_start = period_start.replace(hour=start_hour)
                    period_end = period_end.replace(hour=end_hour, minute=59, second=59)
                    if period_start > period_end:
                        raise ValueError("The selected start date and hour cannot be after the end date and hour.")
                else:
                    period_end = period_end.replace(hour=23, minute=59, second=59)
                form_data["start_year"] = str(period_start.year)
                form_data["end_year"] = str(period_end.year)
                start_date = period_start.strftime("%Y-%m-%d %H:%M:%S")
                end_date = period_end.strftime("%Y-%m-%d %H:%M:%S")
            common = {
                **form_data,
                "dataset": dataset,
                "variables": selected_variables,
                "resolutions": selected_resolutions,
                "seasons": selected_seasons,
                "start_date": start_date,
                "end_date": end_date,
                "show_ocean": False,
                "show_lakes": form_data.get("show_lakes") == "1",
                "show_rivers": False,
                "include_plot": True if service == "analysis" else form_data.get("include_plot") == "1",
            }

            if service == "extraction":
                result = generate_multi_extraction(common, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                result.update({
                    "product_kind": "extraction",
                    "excel_url": url_for("product_download", filename=_relative_export_path(Path(result["excel_path"]))),
                })
                flash("The selected data were extracted successfully.", "success")

            elif service == "visual":
                plot_type = str(form_data.get("visual_type") or "time_series")
                if plot_type not in visual_products:
                    raise ValueError("Select a valid visual product.")
                if plot_type not in dataset_allowed_plots(dataset):
                    raise ValueError("The selected plot is not relevant to this dataset.")
                if plot_type in {"multi_line", "scatter"} and len(selected_variables) < 2:
                    raise ValueError("Select at least two weather elements for the selected comparison plot.")
                if plot_type == "scatter" and len(selected_variables) != 2:
                    raise ValueError("Select exactly two weather elements for a relationship scatter plot.")
                if plot_type == "wind_rose" and dataset != "era5_wind":
                    raise ValueError("Wind Rose is available only for ERA5 Wind Speed and Direction 10m.")
                if plot_type in {"area", "cumulative_total"} and PRODUCT_DATASETS[dataset].get("family") != "rainfall":
                    raise ValueError("Cumulative precipitation plots are available only for precipitation datasets.")
                if plot_type in {"standard_deviation", "coefficient_variation", "mean_std_band", "std_error_bars", "spatial_std_map", "spatial_cv_map"} and "wind_direction" in selected_variables:
                    raise ValueError("Use Wind Speed rather than Wind Direction for standard-deviation variability plots.")

                map_products = {"spatial_map", "spatial_std_map", "spatial_cv_map", "spatial_climatology", "spatial_monthly_climatology", "spatial_annual_series", "spatial_seasonal_climatology"}
                multi_products = {"multi_line", "scatter", "wind_rose", "temperature_monthly_profile"}
                jobs = [(None, None)] if plot_type in map_products | {"heatmap"} else selected_season_jobs(selected_resolutions, selected_seasons)
                with product_data_cache():
                    if plot_type in multi_products:
                        for resolution, season in jobs:
                            params = {
                                **common,
                                "plot_type": plot_type,
                                "variables": selected_variables,
                                "variable": selected_variables[0] if selected_variables else "auto",
                                "resolution": resolution or "monthly",
                                "season": season,
                            }
                            item = generate_plot_product(params, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                            title, subtitle = _professional_product_heading(params, item)
                            if plot_type in {"multi_line", "scatter"} and not str(params.get("custom_plot_title") or "").strip():
                                chosen = [multi_variable_label(dataset, value) for value in selected_variables]
                                title = f"{dict(PLOT_TYPES).get(plot_type, 'Comparison Plot')}: {' and '.join(chosen)} for {form_data.get('location_name')}"
                            item.update({
                                "product_kind": "visual",
                                "product_title": title,
                                "product_subtitle": subtitle,
                                "product_description": PRODUCT_DESCRIPTIONS.get(plot_type, ""),
                                "plot_url": url_for("product_download", filename=_relative_export_path(Path(item["plot_path"]))) if item.get("plot_path") else None,
                                "excel_url": url_for("product_download", filename=_relative_export_path(Path(item["excel_path"]))) if item.get("excel_path") else None,
                            })
                            result_items.append(item)
                    else:
                        for variable in selected_variables:
                            for resolution, season in jobs:
                                params = {
                                    **common,
                                    "plot_type": plot_type,
                                    "variable": variable,
                                    "resolution": resolution or "monthly",
                                    "season": season,
                                }
                                item = generate_plot_product(params, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                                title, subtitle = _professional_product_heading(params, item)
                                item.update({
                                    "product_kind": "visual",
                                    "product_title": title,
                                    "product_subtitle": subtitle,
                                    "product_description": PRODUCT_DESCRIPTIONS.get(plot_type, ""),
                                    "plot_url": url_for("product_download", filename=_relative_export_path(Path(item["plot_path"]))) if item.get("plot_path") else None,
                                    "excel_url": url_for("product_download", filename=_relative_export_path(Path(item["excel_path"]))) if item.get("excel_path") else None,
                                })
                                result_items.append(item)
                flash(f"{dict(PLOT_TYPES).get(plot_type, 'Visual product')} generated successfully.", "success")

            elif service == "analysis":
                analysis_type = str(form_data.get("analysis_type") or "statistical_summary")
                if analysis_type not in analysis_products:
                    raise ValueError("Select a valid climate analysis.")
                with product_data_cache():
                    if analysis_type == "spatial_climatology_analysis":
                        spatial_products = ["spatial_climatology", "spatial_monthly_climatology", "spatial_annual_series", "spatial_seasonal_climatology"]
                        for variable in selected_variables:
                            for spatial_type in spatial_products:
                                params = {**common, "plot_type": spatial_type, "variable": variable, "resolution": "monthly"}
                                item = generate_plot_product(params, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                                title, subtitle = _professional_product_heading(params, item)
                                item.update({
                                    "product_kind": "analysis",
                                    "product_title": title,
                                    "product_subtitle": subtitle,
                                    "product_description": PRODUCT_DESCRIPTIONS.get(spatial_type, ""),
                                    "plot_url": url_for("product_download", filename=_relative_export_path(Path(item["plot_path"]))) if item.get("plot_path") else None,
                                    "excel_url": url_for("product_download", filename=_relative_export_path(Path(item["excel_path"]))) if item.get("excel_path") else None,
                                })
                                result_items.append(item)
                    else:
                        for variable in selected_variables:
                            for resolution, season in selected_season_jobs(selected_resolutions, selected_seasons):
                                params = {
                                    **common,
                                    "plot_type": analysis_type,
                                    "analysis_scope": analysis_type,
                                    "include_plot": True,
                                    "variable": variable,
                                    "resolution": resolution,
                                    "season": season,
                                }
                                item = generate_analysis_bundle(params, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                                item.update({
                                    "product_kind": "analysis",
                                    "excel_url": url_for("product_download", filename=_relative_export_path(Path(item["excel_path"]))) if item.get("excel_path") else None,
                                    "preview_urls": [
                                        url_for("product_download", filename=_relative_export_path(Path(path)))
                                        for path in item.get("preview_paths", [])
                                    ],
                                })
                                result_items.append(item)
                flash(f"{dict(PLOT_TYPES).get(analysis_type, 'Climate analysis')} completed successfully.", "success")

            else:
                index_type = str(form_data.get("index_type") or "total_rainfall")
                valid_indices = dict(RAINFALL_INDICES + TEMPERATURE_INDICES + OTHER_INDICES)
                if index_type not in valid_indices:
                    raise ValueError("Select a valid climate index.")
                allowed_dataset_indices = dict(INDEX_OPTIONS_BY_DATASET.get(dataset, []))
                if index_type not in allowed_dataset_indices:
                    raise ValueError("The selected climate index is not available for this dataset.")
                index_dataset = dataset
                index_resolution = str(form_data.get("index_resolution") or "annual").lower()
                allowed_index_resolutions = INDEX_RESOLUTION_RULES.get(index_type, ["annual"])
                if index_resolution not in allowed_index_resolutions:
                    raise ValueError("The selected time scale is not relevant to this climate index.")
                if index_resolution == "hourly" and index_dataset == "chirps_rainfall":
                    raise ValueError("Hourly precipitation indices require ERA5 Precipitation; CHIRPS starts at daily resolution.")
                index_plot_type = str(form_data.get("index_plot_type") or "auto").lower()
                if index_plot_type not in INDEX_PLOT_RULES.get(index_resolution, ["auto", "line"]):
                    raise ValueError("The selected plot is not relevant to this climate-index time scale.")
                params = {
                    **common,
                    "dataset": index_dataset,
                    "index_type": index_type,
                    "index_resolution": index_resolution,
                    "index_plot_type": index_plot_type,
                    "index_start_date": form_data.get("start_date"),
                    "index_end_date": form_data.get("end_date"),
                    "index_start_hour": form_data.get("start_hour"),
                    "index_end_hour": form_data.get("end_hour"),
                    "index_custom_years": form_data.get("index_custom_years"),
                    "season": form_data.get("index_season"),
                    "custom_months": form_data.get("custom_months"),
                    "include_plot": True,
                }
                item = generate_indices(params, DEFAULT_DATA_DIR, DEFAULT_EXPORT_DIR)
                index_label = valid_indices[index_type]
                resolution_label = INDEX_RESOLUTION_LABELS.get(index_resolution, index_resolution.title())
                period_label = f"{start_date[:10]} to {end_date[:10]}" if index_resolution in {"hourly", "daily"} else f"{form_data.get('start_year')}–{form_data.get('end_year')}"
                item.update({
                    "product_kind": "index",
                    "product_title": str(form_data.get("custom_plot_title") or "").strip() or f"{resolution_label} {index_label} for {form_data.get('location_name', 'Selected Location')}",
                    "product_subtitle": f"{period_label} · {form_data.get('index_season', 'ANNUAL')}",
                    "product_description": "Every generated climate index includes a plot and a downloadable data table.",
                    "excel_url": url_for("product_download", filename=_relative_export_path(Path(item["excel_path"]))) if item.get("excel_path") else None,
                    "plot_url": url_for("product_download", filename=_relative_export_path(Path(item["plot_path"]))) if item.get("plot_path") else None,
                })
                result_items.append(item)
                flash(f"{index_label} calculated successfully.", "success")
        except Exception as exc:
            traceback.print_exc()
            field_errors["generation"] = str(exc)

    if result is not None:
        result_items = [result]

    # A single submission can produce several element/resolution/season
    # outputs. Provide one additional workbook in which every generated output
    # is arranged on its own worksheet, while retaining the individual files.
    generated_workbooks = [
        (str(item.get("product_title") or f"Selection {index}"), Path(item["excel_path"]))
        for index, item in enumerate(result_items, start=1)
        if item.get("excel_path") and Path(item["excel_path"]).exists()
    ]
    if len(generated_workbooks) > 1:
        try:
            combined_dir = Path(DEFAULT_EXPORT_DIR) / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined_path = combined_dir / f"CDE_Multi_Selection_Output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            combine_workbooks_to_separate_sheets(
                combined_path,
                generated_workbooks,
                workbook_title="TMA CDE Multi-Selection Output — Separate Worksheets",
                qr_payload={
                    "document_type": "CDE Multi-Selection Output",
                    "file_name": combined_path.name,
                    "served_by": g.user["full_name"] if g.get("user") else "",
                },
            )
            combined_excel_url = url_for(
                "product_download",
                filename=_relative_export_path(combined_path),
            )
        except Exception:
            traceback.print_exc()
            flash("Individual outputs were generated, but the combined multi-sheet workbook could not be assembled.", "warning")

    return render_template(
        "plots.html",
        sources=sources,
        dataset_sources={key: sources[key] for key in (INDEX_OPTIONS_BY_DATASET if service == "index" else sources)},
        index_options_by_dataset={key: [{"value": value, "label": label} for value, label in values] for key, values in INDEX_OPTIONS_BY_DATASET.items()},
        plot_types=PLOT_TYPES,
        seasons=PRODUCT_SEASONS,
        rainfall_indices=RAINFALL_INDICES,
        temperature_indices=TEMPERATURE_INDICES,
        other_indices=OTHER_INDICES,
        result=result,
        result_items=result_items,
        combined_excel_url=combined_excel_url,
        field_errors=field_errors,
        form_data=form_data,
        active_service=service,
        variable_options=MULTI_VARIABLE_OPTIONS,
        resolution_labels=MULTI_RESOLUTION_LABELS,
        selected_variables=selected_variables,
        selected_resolutions=selected_resolutions,
        selected_seasons=selected_seasons,
        index_resolution_rules=INDEX_RESOLUTION_RULES,
        index_resolution_labels=INDEX_RESOLUTION_LABELS,
        index_plot_types=INDEX_PLOT_TYPES,
        index_plot_rules=INDEX_PLOT_RULES,
    )


@app.route("/parquet-records")
@login_required
def parquet_records():
    abort(404)


@app.route("/parquet-records/download")
@login_required
def parquet_record_download():
    abort(404)


@app.route("/parquet-records/pdf")
@login_required
def parquet_record_pdf():
    abort(404)

@app.route("/verify/<download_id>")
def verify_download(download_id: str):
    row = query_one(
        """
        SELECT e.*, u.full_name AS user_name, u.station_name AS user_station_name,
               COALESCE(e.customer_name, c.customer_name) AS log_customer_name,
               COALESCE(e.customer_organization, c.organization) AS log_customer_organization,
               COALESCE(e.customer_phone, c.phone) AS log_customer_phone,
               COALESCE(e.customer_email, c.email) AS log_customer_email,
               COALESCE(e.customer_address, trim(coalesce(c.postal_address, '') || CASE WHEN c.physical_address IS NOT NULL AND trim(c.physical_address) <> '' THEN '; ' || c.physical_address ELSE '' END)) AS log_customer_address,
               COALESCE(e.cost_recovery_fee, c.cost_recovery_fee) AS log_cost_recovery_fee,
               COALESCE(e.customer_remarks, c.remarks) AS log_customer_remarks
        FROM export_logs e
        LEFT JOIN users u ON u.id = e.user_id
        LEFT JOIN customers c ON c.id = e.customer_id
        WHERE e.download_id = ? AND e.status = 'success'
        ORDER BY e.id DESC
        LIMIT 1
        """,
        (download_id,),
    )
    if not row:
        abort(404)
    payload = {}
    try:
        payload = json.loads(row["qr_payload"] or "{}")
    except Exception:
        payload = {}
    plain_text = qr_payload_to_plain_text(normalize_receipt_payload(row, payload))
    return render_template("verify.html", row=row, payload=payload, plain_text=plain_text)


def _json_list_to_label(value: Any) -> str:
    """Convert a JSON list or Python list into a readable comma-separated label."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    text = str(value).strip()
    if not text:
        return ""
    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            return ", ".join(str(item) for item in decoded if item not in (None, ""))
    except Exception:
        pass
    return text


def normalize_receipt_payload(row: sqlite3.Row, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create one plain-text payload for Data Delivery Report/verification outputs."""
    file_name = payload.get("file_name") or row_value(row, "output_file", "")
    download_id = payload.get("download_id") or row_value(row, "download_id", "")
    data_url = payload.get("data_url") or (make_data_url(file_name) if file_name else "")
    verification_url = payload.get("verification_url") or (make_verification_url(download_id) if download_id else "")

    element = payload.get("element") or payload.get("weather_element")
    if not element and isinstance(payload.get("elements"), list):
        element = ", ".join([str(x) for x in payload.get("elements", []) if x])
    if not element:
        element = _json_list_to_label(row_value(row, "variables", ""))

    data_type = payload.get("data_type")
    if not data_type and isinstance(payload.get("data_types"), list):
        data_type = ", ".join([str(x) for x in payload.get("data_types", []) if x])
    if not data_type:
        data_type = _json_list_to_label(row_value(row, "frequencies", ""))

    customer_name = (
        payload.get("customer_name")
        or row_value(row, "log_customer_name", "")
        or row_value(row, "customer_name", "")
    )
    issued_by = payload.get("issued_by") or row_value(row, "issued_by", "") or row_value(row, "user_name", "")

    return {
        "institution": payload.get("institution") or "Climate Data Extractor",
        "document_type": payload.get("document_type") or "Data Delivery Report",
        "system": payload.get("system") or "Climate Data Extractor",
        "download_id": download_id,
        "customer_name": customer_name,
        "customer_organization": payload.get("customer_organization") or row_value(row, "log_customer_organization", "") or row_value(row, "customer_organization", ""),
        "customer_phone": payload.get("customer_phone") or row_value(row, "log_customer_phone", "") or row_value(row, "customer_phone", ""),
        "customer_email": payload.get("customer_email") or row_value(row, "log_customer_email", "") or row_value(row, "customer_email", ""),
        "customer_address": payload.get("customer_address") or row_value(row, "log_customer_address", "") or row_value(row, "customer_address", ""),
        "cost_recovery_fee": payload.get("cost_recovery_fee") or row_value(row, "log_cost_recovery_fee", "") or row_value(row, "cost_recovery_fee", ""),
        "customer_remarks": payload.get("customer_remarks") or row_value(row, "log_customer_remarks", "") or row_value(row, "customer_remarks", ""),
        "station_name": payload.get("station_name") or row_value(row, "location_name", ""),
        "station_id": payload.get("station_id") or "",
        "latitude": payload.get("latitude") if payload.get("latitude") is not None else row_value(row, "latitude", ""),
        "longitude": payload.get("longitude") if payload.get("longitude") is not None else row_value(row, "longitude", ""),
        "source": payload.get("source") or row_value(row, "source", ""),
        "element": element or "",
        "data_type": data_type or "",
        "start_date": payload.get("start_date") or row_value(row, "start_date", ""),
        "end_date": payload.get("end_date") or row_value(row, "end_date", ""),
        "units": payload.get("units") or "",
        "served_by": issued_by,
        "downloaded_by": issued_by,
        "issued_by": issued_by,
        "user_station": payload.get("user_station") or row_value(row, "user_station_name", ""),
        "downloaded_at": payload.get("downloaded_at") or row_value(row, "created_at", ""),
        "file_name": file_name,
        "data_url": data_url,
        "verification_url": verification_url,
    }

@app.post("/delivery-report/preview")
@login_required
def preview_delivery_report():
    """Preview a draft Data Delivery Report before generating the Excel file."""
    validate_csrf()
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    customer_id = str(request.form.get("customer_id") or "").strip()
    customer = get_customer(customer_id)
    if not customer or int(row_value(customer, "status", 0)) != 1:
        return "Select a registered active customer before previewing the report.", 400

    saved = parse_multi_extraction_requests(request.form.get("requests_json", ""))
    if not saved:
        source_key = str(request.form.get("source") or "").strip()
        cfg = catalog.get(source_key)
        variables = [v for v in request.form.getlist("variables") if cfg and v in cfg.get("variables", {})]
        frequencies = [v for v in request.form.getlist("frequencies") if cfg and v in cfg.get("supported_frequencies", [])]
        name = str(request.form.get("location_name") or "").strip()
        try:
            latitude = float(request.form.get("latitude"))
            longitude = float(request.form.get("longitude"))
        except (TypeError, ValueError):
            latitude = longitude = None
        if not cfg or not variables or not frequencies or not name or latitude is None or longitude is None:
            return "Complete dataset, weather element, resolution, dates, location and coordinates before previewing the report.", 400
        saved = [{
            "source": source_key,
            "variables": variables,
            "frequencies": frequencies,
            "seasons": request.form.getlist("seasons"),
            "start_date": str(request.form.get("start_date") or ""),
            "end_date": str(request.form.get("end_date") or ""),
            "locations": [{"name": name, "latitude": latitude, "longitude": longitude}],
        }]

    request_lines: list[str] = []
    stations: list[str] = []
    elements: list[str] = []
    data_types: list[str] = []
    sources_used: list[str] = []
    for number, item in enumerate(saved, start=1):
        cfg = catalog.get(item.get("source"), {})
        source_label = str(cfg.get("label") or item.get("source") or "")
        variable_labels = [str(cfg.get("variables", {}).get(v, {}).get("label") or v) for v in item.get("variables", [])]
        location_labels = [str(loc.get("name") or "") for loc in item.get("locations", []) if loc.get("name")]
        frequency_labels = [data_type_label(freq) for freq in item.get("frequencies", [])]
        sources_used.append(source_label)
        elements.extend(variable_labels)
        stations.extend(location_labels)
        data_types.extend(frequency_labels)
        request_lines.append(
            f"Request {number}: {source_label}; {', '.join(variable_labels)}; "
            f"{', '.join(frequency_labels)}; {', '.join(location_labels)}; "
            f"{item.get('start_date','')} to {item.get('end_date','')}"
        )

    first = saved[0]
    first_location = (first.get("locations") or [{}])[0]
    preview_id = "DRAFT-PREVIEW"
    payload = {
        "institution": "Climate Data Extractor",
        "document_type": "Draft Data Delivery Report Preview",
        "download_id": preview_id,
        "request_no": preview_id,
        "customer_name": customer_display_name(customer),
        "customer_organization": row_value(customer, "organization", ""),
        "customer_phone": row_value(customer, "phone", ""),
        "customer_email": row_value(customer, "email", ""),
        "customer_address": customer_address_text(customer),
        "source": "; ".join(dict.fromkeys(sources_used)),
        "element": "; ".join(dict.fromkeys(elements)),
        "data_type": ", ".join(dict.fromkeys(data_types)),
        "station_name": "; ".join(dict.fromkeys(stations)),
        "latitude": first_location.get("latitude", ""),
        "longitude": first_location.get("longitude", ""),
        "start_date": first.get("start_date", ""),
        "end_date": first.get("end_date", ""),
        "served_by": g.user["full_name"] if g.get("user") else "",
        "issued_by": g.user["full_name"] if g.get("user") else "",
        "request_details": " | ".join(request_lines),
        "file_name": "Preview only — Excel not generated",
    }
    pdf_bytes = build_receipt_pdf({}, payload)
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=False,
        download_name="CDE_Data_Delivery_Report_Preview.pdf",
        max_age=0,
    )


def build_receipt_pdf(row: sqlite3.Row, payload: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import stringWidth

    receipt_payload = normalize_receipt_payload(row, payload)
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 18 * mm
    now = now_eat()

    def draw_header():
        c.setFont("Times-BoldItalic", 22)
        c.drawCentredString(width / 2, height - 25 * mm, "Data Delivery Report")
        c.setStrokeColor(colors.HexColor("#14A5D5"))
        c.setLineWidth(2.2)
        c.line(0, height - 43 * mm, width, height - 43 * mm)

    def draw_footer():
        c.setStrokeColor(colors.HexColor("#14A5D5"))
        c.setLineWidth(2.2)
        c.line(0, 36 * mm, width, 36 * mm)
        c.setFont("Helvetica-Oblique", 7.8)
        c.drawCentredString(width / 2, 24 * mm, "Generated by the Climate Data Extractor")
        c.drawCentredString(width / 2, 18 * mm, "Verify this report using the embedded QR code or verification URL.")

    def draw_label_value(label, value, y, bold_value=False):
        c.setFont("Times-Roman", 11)
        c.drawString(margin, y, label)
        c.drawString(margin + 47 * mm, y, ":")
        c.setFont("Times-Bold" if bold_value else "Times-Roman", 11)
        c.drawString(margin + 52 * mm, y, str(value or "")[:90])
        return y - 8 * mm

    draw_header()
    y = height - 54 * mm
    ref_no = "CD533/620/01"
    request_no = str(receipt_payload.get("download_id") or row_value(row, "download_id", "")).replace("CDE-", "")[:12]
    c.setFont("Times-Bold", 12)
    c.drawString(margin, y, "In reply please quote:")
    y -= 6 * mm
    c.drawString(margin, y, f"Ref. No. {ref_no}")
    c.drawRightString(width - margin, y - 2 * mm, now.strftime("%d %B, %Y"))
    y -= 14 * mm

    customer_name = receipt_payload.get("customer_name") or receipt_payload.get("customer_organization") or ""
    y = draw_label_value("Request No. (yymmno)", request_no, y)
    y = draw_label_value("Customer Name", customer_name, y)
    y = draw_label_value("Customer Address", receipt_payload.get("customer_address"), y)
    y = draw_label_value("Phone number", receipt_payload.get("customer_phone"), y)
    y = draw_label_value("Email Address", receipt_payload.get("customer_email"), y)

    y -= 2 * mm
    box_top = y
    box_h = 72 * mm
    box_x = margin - 2 * mm
    box_w = width - 2 * box_x
    box_right = box_x + box_w
    inner_x = box_x + 3 * mm
    inner_right = box_right - 3 * mm

    c.setFont("Times-Roman", 10.5)
    c.drawCentredString(width / 2, box_top + 1.5 * mm, "Description for data provided")
    c.setLineWidth(0.8)
    c.setStrokeColor(colors.black)
    c.rect(box_x, box_top - box_h, box_w, box_h, stroke=1, fill=0)

    y = box_top - 10 * mm
    element = receipt_payload.get("element") or "Meteorological Data"
    data_type = receipt_payload.get("data_type") or "Data Product"
    station = receipt_payload.get("station_name") or "Selected Location"
    period = f"{receipt_payload.get('start_date') or ''} to {receipt_payload.get('end_date') or ''}".strip(" to")
    mode = "Electronic copy"

    def _split_long_token(token: str, font: str, size: float, max_width: float) -> list[str]:
        """Split a single token only when it is wider than the available PDF width."""
        if stringWidth(token, font, size) <= max_width:
            return [token]
        pieces: list[str] = []
        current = ""
        for char in token:
            candidate = current + char
            if current and stringWidth(candidate, font, size) > max_width:
                pieces.append(current)
                current = char
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces or [token]

    def _wrap_to_width(text: Any, font: str, size: float, max_width: float) -> list[str]:
        """Wrap text using ReportLab's actual font measurements, not character counts."""
        wrapped: list[str] = []
        paragraphs = str(text or "").replace("\r", "").split("\n")
        for paragraph in paragraphs:
            words: list[str] = []
            for raw_word in paragraph.split():
                words.extend(_split_long_token(raw_word, font, size, max_width))
            if not words:
                wrapped.append("")
                continue
            line = words[0]
            for word in words[1:]:
                candidate = f"{line} {word}" if line else word
                if stringWidth(candidate, font, size) <= max_width:
                    line = candidate
                else:
                    wrapped.append(line)
                    line = word
            wrapped.append(line)
        return wrapped

    def _fit_ellipsis(line: str, font: str, size: float, max_width: float) -> str:
        suffix = "..."
        value = str(line or "")
        while value and stringWidth(value + suffix, font, size) > max_width:
            value = value[:-1]
        return value.rstrip() + suffix

    def draw_wrapped(
        text: Any,
        x: float,
        y_pos: float,
        max_width: float,
        font: str = "Times-Bold",
        size: float = 11.0,
        leading: float = 5.0 * mm,
        max_lines: int | None = None,
        center: bool = False,
    ) -> float:
        lines = _wrap_to_width(text, font, size, max_width)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = _fit_ellipsis(lines[-1], font, size, max_width)
        c.setFont(font, size)
        centre_x = x + max_width / 2
        for line in lines:
            if center:
                c.drawCentredString(centre_x, y_pos, line)
            else:
                c.drawString(x, y_pos, line)
            y_pos -= leading
        return y_pos

    # Parameter text is wrapped within the exact remaining width of the box.
    parameter_label = "Parameter(s) provided: -"
    parameter_value_x = inner_x + 48 * mm
    c.setFont("Times-Bold", 11.0)
    c.drawString(inner_x, y, parameter_label)
    y = draw_wrapped(
        f"{element} - {data_type}",
        parameter_value_x,
        y,
        max_width=inner_right - parameter_value_x,
        font="Times-Bold",
        size=11.0,
        leading=5.0 * mm,
        max_lines=4,
    )

    y -= 1.5 * mm
    c.setFont("Times-Roman", 11.2)
    c.drawString(inner_x, y, "Station (s) provided:-")
    y -= 7.5 * mm
    y = draw_wrapped(
        station,
        inner_x,
        y,
        max_width=inner_right - inner_x,
        font="Times-Bold",
        size=11.0,
        leading=5.0 * mm,
        max_lines=2,
        center=True,
    )

    y -= 2 * mm
    c.setFont("Times-Roman", 11.0)
    c.drawString(inner_x, y, "Period:")
    period_x = inner_x + 18 * mm
    y = draw_wrapped(
        period,
        period_x,
        y,
        max_width=inner_right - period_x,
        font="Times-Bold",
        size=11.0,
        leading=5.0 * mm,
        max_lines=2,
    )

    y -= 3.5 * mm
    c.setFont("Times-Roman", 11.0)
    c.drawString(inner_x, y, "Mode of Delivery:")
    mode_x = inner_x + 39 * mm
    draw_wrapped(
        mode,
        mode_x,
        y,
        max_width=inner_right - mode_x,
        font="Times-Bold",
        size=11.0,
        leading=5.0 * mm,
        max_lines=1,
    )
    y = box_top - box_h - 12 * mm
    served_by = receipt_payload.get("served_by") or receipt_payload.get("issued_by") or ""
    c.setFont("Times-Roman", 12)
    c.drawCentredString(width / 2, y, f"Attended by:- {served_by[:60]}")

    qr_payload = {
        **receipt_payload,
        "reference_no": ref_no,
        "request_no": request_no,
        "period": period,
        "mode_of_delivery": mode,
        "served_by": served_by,
        "date": now.strftime("%d %B, %Y"),
    }
    qr_path = PROJECT_ROOT / "storage" / "exports" / "_tmp_delivery_qr.png"
    qr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        make_qr_png(qr_payload, qr_path)
        qr_size = 52 * mm
        qr_y = max(39 * mm, y - 58 * mm)
        c.drawImage(ImageReader(str(qr_path)), (width - qr_size) / 2, qr_y, width=qr_size, height=qr_size, preserveAspectRatio=True, mask='auto')
        c.setFont("Helvetica", 8)
        c.drawCentredString(width / 2, qr_y - 3 * mm, "Scan QR to verify data delivery details")
    except Exception:
        pass
    draw_footer()

    c.showPage()
    draw_header()
    c.setFont("Times-Bold", 12)
    c.drawCentredString(width / 2, height - 53 * mm, "Thank you for using meteorological data.")

    # Data use and sharing conditions - presented on a dedicated second page,
    # matching the official revised Data Delivery Report layout.
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    conditions_box_x = 29 * mm
    conditions_box_y = 124 * mm
    conditions_box_w = width - (58 * mm)
    conditions_box_h = 109 * mm
    conditions_box_top = conditions_box_y + conditions_box_h

    c.setFillColor(colors.HexColor("#F5FAFC"))
    c.setStrokeColor(colors.HexColor("#14A5D5"))
    c.setLineWidth(1.25)
    c.roundRect(
        conditions_box_x,
        conditions_box_y,
        conditions_box_w,
        conditions_box_h,
        3 * mm,
        stroke=1,
        fill=1,
    )

    c.setFillColor(colors.black)
    c.setFont("Times-Bold", 13)
    c.drawCentredString(
        width / 2,
        conditions_box_top - 12 * mm,
        "DATA USE AND SHARING CONDITIONS",
    )
    c.setStrokeColor(colors.HexColor("#14A5D5"))
    c.setLineWidth(0.9)
    c.line(
        conditions_box_x + 12 * mm,
        conditions_box_top - 17 * mm,
        conditions_box_x + conditions_box_w - 12 * mm,
        conditions_box_top - 17 * mm,
    )

    condition_style = ParagraphStyle(
        "DeliveryConditions",
        fontName="Times-Roman",
        fontSize=10.2,
        leading=13.0,
        textColor=colors.black,
        alignment=0,
        spaceAfter=6.5 * mm,
        allowWidows=0,
        allowOrphans=0,
    )

    conditions = [
        (
            "1. Approved purpose.",
            "Data provided under this report shall be used only for the purpose "
            "stated in the approved data request.",
        ),
        (
            "2. Restriction on sharing.",
            "The recipient shall not share, transfer, reproduce, redistribute, "
            "resell, publish, upload or otherwise make the data available to any third "
            "party without prior written consent from the Tanzania Meteorological "
            "Authority (TMA).",
        ),
        (
            "3. Legal basis and penalties.",
            "This condition is imposed in accordance with section 30(2) of the "
            "Climate Data Extractor Act, 2019. Under section 47, distributing "
            "meteorological data without TMA consent is an offence punishable by a fine "
            "of TZS 20,000,000 to TZS 30,000,000, imprisonment for five to ten years, "
            "or both.",
        ),
        (
            "4. Acknowledgement and further use.",
            "Where the data are used in an approved report, study, research publication "
            "or derived product, TMA shall be acknowledged as the source. Any use beyond "
            "the originally approved purpose requires prior written authorization from TMA.",
        ),
    ]

    paragraph_x = conditions_box_x + 9 * mm
    paragraph_w = conditions_box_w - 18 * mm
    paragraph_y = conditions_box_top - 24 * mm
    for heading, body in conditions:
        paragraph = Paragraph(f"<b>{heading}</b> {body}", condition_style)
        paragraph_h = paragraph.wrap(paragraph_w, conditions_box_h)[1]
        paragraph_y -= paragraph_h
        paragraph.drawOn(c, paragraph_x, paragraph_y)
        paragraph_y -= condition_style.spaceAfter

    # Repeat the report verification QR on every page of the Data Delivery Report.
    try:
        if qr_path.exists():
            page_qr_size = 52 * mm
            c.drawImage(
                ImageReader(str(qr_path)),
                width - 70 * mm,
                53 * mm,
                width=page_qr_size,
                height=page_qr_size,
                preserveAspectRatio=True,
                mask='auto',
            )
            c.setFont("Helvetica", 6.8)
            c.drawCentredString(width - 44 * mm, 49 * mm, "QR verification")
    except Exception:
        pass
    draw_footer()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


@app.route("/receipt/<download_id>")
def download_receipt(download_id: str):
    row = query_one(
        """
        SELECT e.*, u.full_name AS user_name, u.station_name AS user_station_name,
               COALESCE(e.customer_name, c.customer_name) AS log_customer_name,
               COALESCE(e.customer_organization, c.organization) AS log_customer_organization,
               COALESCE(e.customer_phone, c.phone) AS log_customer_phone,
               COALESCE(e.customer_email, c.email) AS log_customer_email,
               COALESCE(e.customer_address, trim(coalesce(c.postal_address, '') || CASE WHEN c.physical_address IS NOT NULL AND trim(c.physical_address) <> '' THEN '; ' || c.physical_address ELSE '' END)) AS log_customer_address,
               COALESCE(e.cost_recovery_fee, c.cost_recovery_fee) AS log_cost_recovery_fee,
               COALESCE(e.customer_remarks, c.remarks) AS log_customer_remarks
        FROM export_logs e
        LEFT JOIN users u ON u.id = e.user_id
        LEFT JOIN customers c ON c.id = e.customer_id
        WHERE e.download_id = ? AND e.status = 'success'
        ORDER BY e.id DESC
        LIMIT 1
        """,
        (download_id,),
    )
    if not row:
        abort(404)
    try:
        payload = json.loads(row["qr_payload"] or "{}")
    except Exception:
        payload = {}
    pdf_bytes = build_receipt_pdf(row, payload)
    response = send_file(
        BytesIO(pdf_bytes),
        as_attachment=(request.args.get("inline") != "1"),
        download_name=make_cde_receipt_filename(download_id, row["created_at"] if "created_at" in row.keys() else None),
        mimetype="application/pdf",
        max_age=0,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/downloads/<path:filename>")
@login_required
def download_file(filename: str):
    safe_name = secure_filename(filename)
    path = DEFAULT_EXPORT_DIR / safe_name
    if path.suffix.lower() == ".csv" or not path.exists():
        abort(404)
    response = send_file(
        path,
        as_attachment=True,
        download_name=safe_name,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        max_age=0,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response




@app.route("/users", methods=["GET", "POST"])
@admin_required
def users():
    field_errors: dict[str, str] = {}
    form_data: dict[str, str] = {"full_name": "", "email": "", "phone": "", "role": "user"}
    if request.method == "POST":
        validate_csrf()
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        role = request.form.get("role", "user")
        form_data = {"full_name": full_name, "email": email, "phone": phone, "role": role}

        if not full_name:
            field_errors["fullNameField"] = "Full name is required."
        if not email:
            field_errors["emailField"] = "Email address is required."
        if role not in {"admin", "user"}:
            field_errors["roleField"] = "Select a valid role."

        if not field_errors:
            try:
                otp = generate_otp()
                now = datetime.now().isoformat(timespec="seconds")
                expires = (datetime.now() + timedelta(days=7)).isoformat(timespec="seconds")
                placeholder_password = "CDE@" + secrets.token_urlsafe(12)
                get_db().execute(
                    """
                    INSERT INTO users (full_name, email, phone, role, password_hash, status, force_password_change, otp, otp_generated_at, otp_expires_at)
                    VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, ?)
                    """,
                    (full_name, email, phone, role, generate_password_hash(placeholder_password), otp, now, expires),
                )
                get_db().commit()
                flash(f"User created successfully. Generated OTP for {email}: {otp}", "success")
                return redirect(url_for("users"))
            except sqlite3.IntegrityError:
                field_errors["emailField"] = "A user with this email already exists."
    rows = query_all("SELECT * FROM users ORDER BY id DESC")
    return render_template("users.html", rows=rows, field_errors=field_errors, form_data=form_data)


@app.route("/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def toggle_user(user_id: int):
    validate_csrf()
    if int(g.user["id"]) == user_id:
        flash("You cannot disable your own account.", "error")
        return redirect(url_for("users"))
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user:
        abort(404)
    new_status = 0 if user["status"] else 1
    get_db().execute("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (new_status, datetime.now().isoformat(timespec="seconds"), user_id))
    get_db().commit()
    flash("User status updated.", "success")
    return redirect(url_for("users"))


@app.route("/users/<int:user_id>/reset", methods=["POST"])
@admin_required
def admin_reset_user(user_id: int):
    validate_csrf()
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user:
        abort(404)
    otp = generate_otp()
    now = datetime.now().isoformat(timespec="seconds")
    expires = (datetime.now() + timedelta(days=7)).isoformat(timespec="seconds")
    placeholder_password = "CDE@" + secrets.token_urlsafe(12)
    get_db().execute(
        """
        UPDATE users
        SET password_hash = ?, otp = ?, otp_generated_at = ?, otp_expires_at = ?, force_password_change = 1, updated_at = ?
        WHERE id = ?
        """,
        (generate_password_hash(placeholder_password), otp, now, expires, now, user_id),
    )
    get_db().commit()
    flash(f"New OTP for {user['email']}: {otp}", "success")
    return redirect(url_for("users"))


@app.route("/users/<int:user_id>/view")
@admin_required
def view_user(user_id: int):
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user:
        abort(404)
    exports = query_all(
        """
        SELECT * FROM export_logs
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 10
        """,
        (user_id,),
    )
    return render_template("user_view.html", row=user, exports=exports)


@app.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_user(user_id: int):
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user:
        abort(404)
    if request.method == "POST":
        validate_csrf()
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        role = request.form.get("role", "user")
        status = 1 if request.form.get("status") == "1" else 0
        if not full_name or not email:
            flash("Full name and email are required.", "error")
        elif role not in {"admin", "user"}:
            flash("Invalid role selected.", "error")
        elif int(g.user["id"]) == user_id and status == 0:
            flash("You cannot deactivate your own account.", "error")
        else:
            try:
                now = datetime.now().isoformat(timespec="seconds")
                get_db().execute(
                    """
                    UPDATE users
                    SET full_name = ?, email = ?, phone = ?, role = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (full_name, email, phone, role, status, now, user_id),
                )
                get_db().commit()
                flash("User details updated successfully.", "success")
                return redirect(url_for("users"))
            except sqlite3.IntegrityError:
                flash("Another user already uses that email address.", "error")
    return render_template("user_edit.html", row=user)


@app.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id: int):
    validate_csrf()
    if int(g.user["id"]) == user_id:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("users"))
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user:
        abort(404)
    get_db().execute("DELETE FROM users WHERE id = ?", (user_id,))
    get_db().commit()
    flash(f"User {user['email']} deleted successfully.", "success")
    return redirect(url_for("users"))


@app.route("/health")
def health():
    return {"status": "ok", "system": APP_SHORT_NAME, "database": str(DB_PATH)}


if __name__ == "__main__":
    ensure_dirs()
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
