def weather_full_name(value, dataset=None):
    """
    Convert dataset variable codes into readable weather element names.
    Works for Export Logs, QR text, downloaded files, and templates.
    """
    if value is None:
        return ""

    raw = str(value).strip()
    dataset_key = str(dataset or "").lower()
    # Handle labels already converted from JSON lists, such as "tp, t2m".
    if "," in raw:
        return ", ".join([weather_full_name(part.strip(), dataset) for part in raw.split(",") if part.strip()])
    key = raw.lower()

    combined_map = {
        ("era5_rh", "r"): "Relative Humidity",
        ("era5_rh", "rh"): "Relative Humidity",
        ("chirps", "precip"): "CHIRPS Precipitation",
        ("chirps", "rainfall"): "CHIRPS Precipitation",
        ("era5_tp", "tp"): "ERA5 Precipitation",
        ("era5_tp", "precip"): "ERA5 Precipitation",
        ("era5_tp", "total_precipitation"): "ERA5 Precipitation",
    }

    for ds, code in combined_map:
        if ds in dataset_key and key == code:
            return combined_map[(ds, code)]

    general_map = {
        "r": "Relative Humidity",
        "rh": "Relative Humidity",
        "relative_humidity": "Relative Humidity",

        "precip": "CHIRPS Precipitation",
        "tp": "ERA5 Precipitation",
        "rain": "Rainfall",
        "rainfall": "Rainfall",
        "total_precipitation": "ERA5 Precipitation",

        "t2m": "2 Metre Temperature",
        "2m_temperature": "2 Metre Temperature",
        "temperature": "Temperature",
        "tas": "Mean Temperature",
        "ta": "Mean Temperature",
        "mean_temperature": "Mean Temperature",

        "mx2t": "Maximum 2 Metre Temperature",
        "tnx": "Maximum Temperature",
        "tx": "Maximum Temperature",
        "tmax": "Maximum Temperature",
        "maximum_temperature": "Maximum Temperature",

        "mn2t": "Minimum 2 Metre Temperature",
        "tn": "Minimum Temperature",
        "tmin": "Minimum Temperature",
        "minimum_temperature": "Minimum Temperature",

        "d2m": "2 Metre Dewpoint Temperature",
        "dew_point_temperature": "Dew Point Temperature",

        "skt": "Skin Temperature",
        "stl1": "Soil Temperature Level 1",
        "stl2": "Soil Temperature Level 2",
        "soil_temperature_level_1": "Soil Temperature Level 1",
        "soil_temperature_level_2": "Soil Temperature Level 2",

        "u10": "10 Metre U-Wind Component",
        "v10": "10 Metre V-Wind Component",
        "wind_speed": "Wind Speed",
        "wind_direction": "Wind Direction",

        "sp": "Surface Pressure",
        "msl": "Mean Sea Level Pressure",
        "ssrd": "Surface Solar Radiation Downwards",
        "evaporation": "Evaporation",
        "e": "Evaporation",
        "ro": "Runoff",
        "swvl1": "Volumetric Soil Moisture Layer 1",
        "swvl2": "Volumetric Soil Moisture Layer 2",
    }

    return general_map.get(key, raw.replace("_", " ").title())
