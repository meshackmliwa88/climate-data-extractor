#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="/var/www/html/netcdf_data_extractor"
SERVICE_NAME="netcdf-extractor"
NGINX_SITE="netcdf_data_extractor"
APP_OWNER="cde"
APP_GROUP="www-data"

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this deployment script with sudo." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip nginx rsync unzip build-essential pkg-config

# Create the target and all persistent locations before synchronising code.
# The entire storage folder is excluded from rsync, so no existing climate data,
# database, Parquet, export, invoice, QR or other stored file is deleted.
mkdir -p \
    "$TARGET_DIR/storage/zarr/hourly" \
    "$TARGET_DIR/storage/zarr/daily" \
    "$TARGET_DIR/storage/zarr/dekadal" \
    "$TARGET_DIR/storage/zarr/monthly" \
    "$TARGET_DIR/storage/zarr/annual" \
    "$TARGET_DIR/storage/zarr/seasonal" \
    "$TARGET_DIR/storage/zarr/climate_indices" \
    "$TARGET_DIR/storage/zarr/shapefiles" \
    "$TARGET_DIR/storage/db/parquet" \
    "$TARGET_DIR/storage/exports" \
    "$TARGET_DIR/storage/uploads" \
    "$TARGET_DIR/logs"

rsync -a --delete \
    --exclude='storage/' \
    --exclude='venv/' \
    --exclude='.env' \
    --exclude='logs/' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

cd "$TARGET_DIR"

if [[ ! -x "$TARGET_DIR/venv/bin/python" ]]; then
    rm -rf "$TARGET_DIR/venv"
    python3 -m venv "$TARGET_DIR/venv"
fi

"$TARGET_DIR/venv/bin/python" -m pip install --upgrade pip
"$TARGET_DIR/venv/bin/pip" install -r "$TARGET_DIR/requirements.txt"

# Fail before restarting the live service when application source contains a syntax error.
"$TARGET_DIR/venv/bin/python" -m compileall -q \
    "$TARGET_DIR/app.py" \
    "$TARGET_DIR/cde_store.py" \
    "$TARGET_DIR/cde_products.py" \
    "$TARGET_DIR/cde_variable_selection.py" \
    "$TARGET_DIR/cde_analysis.py" \
    "$TARGET_DIR/cde_multi.py" \
    "$TARGET_DIR/scripts"

# Run focused production regressions before the live service is restarted.
# These tests cover non-indexed valid_time coordinates, split mean/min/max
# stores and numeric statistic-dimension ordering.
PYTHONPATH="$TARGET_DIR" "$TARGET_DIR/venv/bin/python" -m unittest \
    tests.test_production_log_regressions

# Report whether the preserved lake layer contains the transboundary Lake Nyasa/Lake Malawi.
# This is informational and never blocks deployment.
"$TARGET_DIR/venv/bin/python" "$TARGET_DIR/scripts/check_tanzania_lakes.py" \
    "$TARGET_DIR/storage/zarr" || true

install -m 644 \
    "$TARGET_DIR/deployment/netcdf-extractor.service" \
    "/etc/systemd/system/netcdf-extractor.service"

install -m 644 \
    "$TARGET_DIR/deployment/nginx_netcdf_extractor.conf" \
    "/etc/nginx/sites-available/$NGINX_SITE"

ln -sfn \
    "/etc/nginx/sites-available/$NGINX_SITE" \
    "/etc/nginx/sites-enabled/$NGINX_SITE"

# Apply the requested full permissions to the complete deployed project,
# including the preserved storage folder and all of its existing contents.
chown -R "$APP_OWNER:$APP_GROUP" "$TARGET_DIR"
chmod -R u=rwX,g=rwX,o=rX "$TARGET_DIR"

nginx -t
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
systemctl reload nginx

# Reapply full permissions after package installation and service preparation.
chown -R "$APP_OWNER:$APP_GROUP" "$TARGET_DIR"
chmod -R u=rwX,g=rwX,o=rX "$TARGET_DIR"

echo
echo "CDE deployment completed."
echo "Persistent storage preserved: $TARGET_DIR/storage"
echo "Service status:"
systemctl --no-pager --full status "$SERVICE_NAME" || true
