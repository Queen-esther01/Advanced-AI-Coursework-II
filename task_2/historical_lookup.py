from pathlib import Path

import pandas as pd

from task_2.utils import (
    DEFAULT_MINUTES_PER_STOP,
    WAT2WEY_STOP_CODES,
    WEY2WAT_STOP_CODES,
    resolve_station_code,
)

LOOKUP_CSV = (
    Path(__file__).parent / "train_historical_data" / "journey_metrics_lookup.csv"
)
_LOOKUP_TABLE = None


def _load_lookup_table():
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is not None:
        return _LOOKUP_TABLE

    if not LOOKUP_CSV.exists():
        raise FileNotFoundError(
            f"Historical journey lookup not found: {LOOKUP_CSV}. "
            "Regenerate it from train_historical_data/*.xlsx."
        )

    table = pd.read_csv(LOOKUP_CSV)
    _LOOKUP_TABLE = {
        (row.journey, row.current_station, row.destination_station): {
            "stops_remaining": int(row.stops_remaining),
            "remaining_minutes": float(row.remaining_minutes),
            "sample_count": int(row.sample_count),
        }
        for row in table.itertuples(index=False)
    }
    return _LOOKUP_TABLE


def _geographic_stops_remaining(journey_key, current_code, destination_code):
    if journey_key == "WEY2WAT":
        stop_codes = WEY2WAT_STOP_CODES
    elif journey_key == "WAT2WEY":
        stop_codes = WAT2WEY_STOP_CODES
    else:
        return None

    if current_code not in stop_codes or destination_code not in stop_codes:
        return None

    current_index = stop_codes.index(current_code)
    destination_index = stop_codes.index(destination_code)
    if current_index >= destination_index:
        return None

    return destination_index - current_index


def lookup_journey_metrics(journey, current_location, destination):
    """
    Returns median stops_remaining and remaining_minutes for a journey segment
    using historical calling patterns from past WEY↔WAT services. Falls back to
    the geographic route list only when no historical pair exists. Returns a dict
    with stops_remaining, remaining_minutes, and source ('historical' or
    'fallback'), or None values when the segment is invalid.
    """
    journey_key = (journey or "").strip().upper()
    current_code = resolve_station_code(current_location)
    destination_code = resolve_station_code(destination)

    if journey_key not in {"WEY2WAT", "WAT2WEY"}:
        return {
            "stops_remaining": None,
            "remaining_minutes": None,
            "source": None,
            "current_station_code": current_code,
            "destination_station_code": destination_code,
        }

    if not current_code or not destination_code:
        return {
            "stops_remaining": None,
            "remaining_minutes": None,
            "source": None,
            "current_station_code": current_code,
            "destination_station_code": destination_code,
        }

    lookup = _load_lookup_table()
    key = (journey_key, current_code, destination_code)
    match = lookup.get(key)
    if match:
        return {
            "stops_remaining": match["stops_remaining"],
            "remaining_minutes": match["remaining_minutes"],
            "source": "historical",
            "sample_count": match["sample_count"],
            "current_station_code": current_code,
            "destination_station_code": destination_code,
        }

    stops_remaining = _geographic_stops_remaining(
        journey_key, current_code, destination_code
    )
    if stops_remaining is None:
        return {
            "stops_remaining": None,
            "remaining_minutes": None,
            "source": None,
            "current_station_code": current_code,
            "destination_station_code": destination_code,
        }

    return {
        "stops_remaining": stops_remaining,
        "remaining_minutes": round(stops_remaining * DEFAULT_MINUTES_PER_STOP, 1),
        "source": "fallback",
        "current_station_code": current_code,
        "destination_station_code": destination_code,
    }
