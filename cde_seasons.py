# Tanzania / climate seasonal definitions for CDE

SEASONS = {
    "DJF": {
        "label": "DJF - December, January, February",
        "months": [12, 1, 2],
        "description": "Hot season / early rainy season depending on region",
    },
    "MAM": {
        "label": "MAM - March, April, May",
        "months": [3, 4, 5],
        "description": "Masika / long rains",
    },
    "JJA": {
        "label": "JJA - June, July, August",
        "months": [6, 7, 8],
        "description": "Cool dry season / Kipupwe",
    },
    "SON": {
        "label": "SON - September, October, November",
        "months": [9, 10, 11],
        "description": "Transition into short rains",
    },
    "OND": {
        "label": "OND - October, November, December",
        "months": [10, 11, 12],
        "description": "Vuli / short rains",
    },
    "NDJ": {
        "label": "NDJ - November, December, January",
        "months": [11, 12, 1],
        "description": "Start of main rainy season in some unimodal areas",
    },
    "DJFMA": {
        "label": "DJFMA - December to April",
        "months": [12, 1, 2, 3, 4],
        "description": "Main rainy season in many unimodal areas",
    },
    "NDJFMA": {
        "label": "NDJFMA - November to April",
        "months": [11, 12, 1, 2, 3, 4],
        "description": "Full unimodal rainy season",
    },
}


def get_season_months(season_code):
    if not season_code:
        return []
    season_code = season_code.upper().strip()
    return SEASONS.get(season_code, {}).get("months", [])


def season_year_from_date(dt, season_code):
    """
    Assign season year correctly for seasons crossing calendar years.
    Example:
    Dec 2024 in DJF belongs to DJF 2025.
    Jan 2025 and Feb 2025 also belong to DJF 2025.
    """
    season_code = season_code.upper().strip()
    month = int(dt.month)
    year = int(dt.year)

    cross_year_seasons = {"DJF", "NDJ", "DJFMA", "NDJFMA"}

    if season_code in cross_year_seasons and month in [11, 12]:
        return year + 1

    return year


def is_sum_variable(variable_name):
    """
    Rainfall/precipitation should be summed.
    Other variables such as temperature, humidity, wind, pressure should be averaged.
    """
    name = str(variable_name).lower()
    sum_keywords = [
        "precipitation",
        "rainfall",
        "rain",
        "tp",
        "total precipitation",
    ]
    return any(k in name for k in sum_keywords)
