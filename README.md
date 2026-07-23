# Climate Data Extractor (CDE)

CDE is a Flask-based platform for extracting, analysing and visualising CHIRPS and ERA5 climate datasets stored in NetCDF and Zarr format. It supports point extraction, climate indices, plots, maps, Excel outputs and QR-based verification.

## Public-build scope

This public package intentionally excludes institutional logos and the following administrative or commercial modules:

- Export Logs
- Data Cost Recovery
- Customers
- Proposed Cost Recovery
- Station Registration

Large datasets, runtime databases, user records and generated files are also excluded.

## Main capabilities

- CHIRPS and ERA5 dataset catalogue
- Hourly, daily, monthly, annual and seasonal extraction
- Climate indices and statistical analysis
- Time-series, climatology, anomaly, variability and spatial products
- Excel and PNG generation
- QR verification without embedded organization logos
- User authentication and administration

## Requirements

- Python 3.11 or newer
- Sufficient memory for the selected climate products
- Local or mounted Zarr datasets matching `config/zarr_catalog.json`

## Install

```bash
git clone <your-public-repository-url>
cd netcdf_data_extractor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
nano .env
set -a
source .env
set +a
```

Use a strong `CDE_SECRET_KEY` and administrator password. Never commit the real `.env` file.

Place the climate stores under `storage/zarr`, or set an external path:

```bash
export CDE_ZARR_DIR=/path/to/zarr
```

The expected structure is:

```text
storage/zarr/
├── hourly/
├── daily/
├── monthly/
├── annual/
├── seasonal/
└── climate_indices/
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

On the first run, CDE creates a local SQLite database under `storage/db`. When `CDE_ADMIN_PASSWORD` is not supplied, a one-time random administrator password is printed to the terminal. Set the environment variable explicitly for controlled deployment.

## Test

```bash
python -m compileall -q .
pytest -q
```

## Repository safety

The `.gitignore` excludes secrets, databases, Zarr/NetCDF files, generated Excel/PDF/PNG outputs and runtime logs. Check staged files before every public push:

```bash
git status
git diff --cached --name-only
```

## Data

Climate datasets are not distributed with this repository. Users are responsible for obtaining datasets under their original licences and configuring the catalogue paths.

## License

No open-source licence is included. Public visibility permits viewing and cloning the repository but does not automatically grant reuse rights.
