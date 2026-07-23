#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  build-essential pkg-config \
  unzip nginx

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

python - <<'PY'
import flask, pandas, xarray, dask, numpy, openpyxl, werkzeug, qrcode, PIL, reportlab, zarr, numcodecs
print('All required Python packages, including Zarr, are working.')
PY

mkdir -p \
  storage/zarr/hourly storage/zarr/daily storage/zarr/monthly \
  storage/zarr/annual storage/zarr/seasonal storage/zarr/climate_indices \
  storage/exports storage/db/parquet storage/uploads logs
chmod -R 775 storage logs

python - <<'PY'
from app import init_db
init_db()
print('CDE database initialized.')
PY

echo ''
echo 'Installation complete.'
echo 'Run: source venv/bin/activate && python app.py'
