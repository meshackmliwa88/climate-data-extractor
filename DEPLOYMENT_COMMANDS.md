# TMA CDE — one-click deployment

This release preserves the complete existing directory:

```text
/var/www/html/netcdf_data_extractor/storage
```

## One command

Place `netcdf_data_extractor_climate_maps_full_update.zip` in `~/Downloads`, then run:

```bash
cd ~/Downloads && rm -rf /tmp/cde_climate_maps_update && mkdir -p /tmp/cde_climate_maps_update && unzip -o netcdf_data_extractor_climate_maps_full_update.zip -d /tmp/cde_climate_maps_update && sudo bash /tmp/cde_climate_maps_update/netcdf_data_extractor/deployment/deploy_preserve_storage.sh
```

The deployment script synchronises application code with `rsync --delete` while explicitly excluding `storage/`. It installs requirements, validates Python source files, applies the requested permissions, restarts `netcdf-extractor`, and reloads Nginx.

## Validate the Zarr catalogue after deployment

```bash
cd /var/www/html/netcdf_data_extractor && source venv/bin/activate && python scripts/check_zarr_mapping.py
```

## Inspect the service

```bash
sudo systemctl status netcdf-extractor --no-pager && sudo journalctl -u netcdf-extractor -n 200 --no-pager
```
