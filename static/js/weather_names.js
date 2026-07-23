function fullWeatherElementName(value, dataset) {
    if (!value) return "";

    const raw = String(value).trim();
    const key = raw.toLowerCase();
    const ds = String(dataset || "").toLowerCase();

    if (ds.includes("era5_rh") && (key === "r" || key === "rh")) return "Relative Humidity";
    if (ds.includes("chirps") && key === "precip") return "CHIRPS Precipitation";
    if (ds.includes("era5_tp") && (key === "tp" || key === "precip")) return "ERA5 Precipitation";

    const map = {
        "r": "Relative Humidity",
        "rh": "Relative Humidity",
        "relative_humidity": "Relative Humidity",

        "precip": "CHIRPS Precipitation",
        "tp": "ERA5 Precipitation",
        "rainfall": "Rainfall",
        "rain": "Rainfall",
        "total_precipitation": "ERA5 Precipitation",

        "t2m": "2 Metre Temperature",
        "2m_temperature": "2 Metre Temperature",
        "temperature": "Temperature",
        "ta": "Mean Temperature",
        "tas": "Mean Temperature",
        "mean_temperature": "Mean Temperature",

        "mx2t": "Maximum 2 Metre Temperature",
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

        "u10": "10 Metre U-Wind Component",
        "v10": "10 Metre V-Wind Component",
        "wind_speed": "Wind Speed",
        "wind_direction": "Wind Direction",

        "sp": "Surface Pressure",
        "msl": "Mean Sea Level Pressure",
        "ssrd": "Surface Solar Radiation Downwards",
        "e": "Evaporation",
        "evaporation": "Evaporation",
        "ro": "Runoff",
        "swvl1": "Volumetric Soil Moisture Layer 1",
        "swvl2": "Volumetric Soil Moisture Layer 2"
    };

    return map[key] || raw.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
}

function updateWeatherElementCells() {
    const tables = document.querySelectorAll("table");

    tables.forEach(table => {
        const headers = Array.from(table.querySelectorAll("thead th, tr:first-child th"));
        if (!headers.length) return;

        const weatherIndex = headers.findIndex(th =>
            th.textContent.trim().toLowerCase().includes("weather element")
        );

        const datasetIndex = headers.findIndex(th =>
            th.textContent.trim().toLowerCase().includes("dataset")
        );

        if (weatherIndex === -1) return;

        const rows = table.querySelectorAll("tbody tr");

        rows.forEach(row => {
            const cells = row.querySelectorAll("td");
            if (!cells.length || !cells[weatherIndex]) return;

            const current = cells[weatherIndex].textContent.trim();
            const dataset = datasetIndex >= 0 && cells[datasetIndex]
                ? cells[datasetIndex].textContent.trim()
                : "";

            const fullName = fullWeatherElementName(current, dataset);

            if (current && current !== fullName) {
                cells[weatherIndex].textContent = fullName;
                cells[weatherIndex].title = current;
            }
        });
    });
}

document.addEventListener("DOMContentLoaded", updateWeatherElementCells);

const observer = new MutationObserver(() => updateWeatherElementCells());
observer.observe(document.body, { childList: true, subtree: true });
