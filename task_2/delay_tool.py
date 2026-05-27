import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from task_2.historical_lookup import lookup_journey_metrics
from task_2.utils import resolve_station_code

MODEL_DIR = Path(__file__).parent / "models"
_MODEL_CACHE = {}


def _parse_time_to_minutes(time_text):
    """
    Parses a clock time such as '14:30', '17:55', or '5:55pm' into minutes since
    midnight. Returns an integer, or None if the format is not recognised.

    Times without am/pm are treated as 24-hour when hour >= 13. For hours 1-6,
    assumes PM (e.g. '5:55' -> 17:55) because passengers often omit am/pm for
    evening commuter services. Hours 7-12 without am/pm are treated as morning/noon.
    """
    time_text = (time_text or "").strip().lower()
    if not time_text:
        return None

    match = re.match(
        r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$",
        time_text.replace(".", ":"),
    )
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem is None:
        if 1 <= hour <= 6:
            hour += 12
        elif hour == 12:
            pass
        elif hour >= 13:
            pass

    if hour > 23 or minute > 59:
        return None

    return hour * 60 + minute


def _load_model_artifact(train_journey):
    """
    Loads and caches the HistGradientBoosting artifact for the given journey
    direction. Returns the joblib dict with model and feature_columns.
    """
    journey_key = (train_journey or "").strip().upper()
    if journey_key not in {"WEY2WAT", "WAT2WEY"}:
        raise ValueError("train_journey must be 'WEY2WAT' or 'WAT2WEY'")

    if journey_key in _MODEL_CACHE:
        return _MODEL_CACHE[journey_key]

    model_path = MODEL_DIR / f"hgb_{journey_key.lower()}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)
    _MODEL_CACHE[journey_key] = artifact
    return artifact


def _build_feature_row(
    artifact,
    current_delay,
    planned_time_now,
    remaining_minutes,
    stops_remaining,
    current_station_code,
):
    """
    Builds a single-row DataFrame matching the trained model's feature columns.
    Returns the feature matrix ready for model.predict.
    """
    station_col = f"station_{current_station_code}"
    row = {
        "current_delay": float(current_delay),
        "planned_time_now": int(planned_time_now),
        "remaining_minutes": float(remaining_minutes),
        "stops_remaining": int(stops_remaining),
        "day_of_week": datetime.now().weekday(),
        "direction_wey2wat": artifact["direction_wey2wat"],
    }

    for col in artifact["feature_columns"]:
        if col.startswith("station_"):
            row[col] = 1 if col == station_col else 0
        elif col not in row:
            row[col] = 0

    return pd.DataFrame([row])[artifact["feature_columns"]]


def get_train_delay(
    train_journey,
    current_location,
    destination,
    current_delay,
    planned_time_at_current_stop,
):
    """
    Runs the direction-specific delay model (WEY2WAT or WAT2WEY) using the
    passenger and journey inputs collected in chat. Stops remaining and
    remaining minutes are derived from the route in code. Returns a dict with
    predicted_delay_minutes and supporting journey details. Required for the
    get_train_delay LLM tool after station coverage is confirmed.
    """
    journey_key = (train_journey or "").strip().upper()
    current_code = resolve_station_code(current_location)
    destination_code = resolve_station_code(destination)

    if not current_code:
        return {"error": f"Could not resolve current location: {current_location}"}
    if not destination_code:
        return {"error": f"Could not resolve destination: {destination}"}

    planned_time_now = _parse_time_to_minutes(planned_time_at_current_stop)
    if planned_time_now is None:
        return {
            "error": (
                "planned_time_at_current_stop must be a valid time such as "
                "'14:30' or '2:30pm'"
            )
        }

    metrics = lookup_journey_metrics(journey_key, current_location, destination)
    stops_remaining = metrics["stops_remaining"]
    remaining_minutes = metrics["remaining_minutes"]
    metrics_source = metrics["source"]

    if stops_remaining is None or stops_remaining < 1:
        return {
            "error": (
                "Could not compute stops_remaining for this journey. Check that "
                "current location is before destination on the route."
            )
        }

    print(
        f"Getting delay for train journey: {journey_key}, "
        f"current location: {current_location} ({current_code}), "
        f"destination: {destination} ({destination_code}), "
        f"current delay: {current_delay}, "
        f"planned time at current stop: {planned_time_at_current_stop}, "
        f"stops remaining: {stops_remaining} ({metrics_source}), "
        f"remaining minutes: {remaining_minutes} ({metrics_source})"
    )

    try:
        artifact = _load_model_artifact(journey_key)
        features = _build_feature_row(
            artifact,
            current_delay=current_delay,
            planned_time_now=planned_time_now,
            remaining_minutes=remaining_minutes,
            stops_remaining=stops_remaining,
            current_station_code=current_code,
        )
        predicted_delay = float(artifact["model"].predict(features)[0])
    except Exception as exc:
        return {"error": f"Delay prediction failed: {exc}"}

    rounded_delay = round(predicted_delay, 1)
    if rounded_delay > 0:
        reason = (
            f"The train is predicted to arrive at {destination} about "
            f"{rounded_delay:g} minutes late."
        )
    elif rounded_delay < 0:
        reason = (
            f"The train is predicted to arrive at {destination} about "
            f"{abs(rounded_delay):g} minutes early."
        )
    else:
        reason = f"The train is predicted to arrive at {destination} on time."

    return {
        "predicted_delay_minutes": rounded_delay,
        "train_journey": journey_key,
        "current_location": current_location,
        "current_station_code": current_code,
        "destination": destination,
        "destination_station_code": destination_code,
        "current_delay_minutes": float(current_delay),
        "planned_time_at_current_stop": planned_time_at_current_stop,
        "stops_remaining": int(stops_remaining),
        "remaining_minutes": float(remaining_minutes),
        "metrics_source": metrics_source,
        "reason": reason,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_train_delay",
            "description": (
                "Predict the expected final delay at the passenger's destination "
                "for a WEY↔WAT service. Call only after check_station_coverage "
                "confirms the journey is supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "train_journey": {
                        "type": "string",
                        "description": "Journey direction: 'WEY2WAT' or 'WAT2WEY'",
                    },
                    "current_location": {
                        "type": "string",
                        "description": "Station where the train currently is",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Passenger destination station",
                    },
                    "current_delay": {
                        "type": "number",
                        "description": (
                            "Current delay in minutes at the present stop "
                            "(negative if early)"
                        ),
                    },
                    "planned_time_at_current_stop": {
                        "type": "string",
                        "description": (
                            "Scheduled departure or arrival time at the current "
                            "stop, e.g. '17:55', '5:55pm', or '14:30'"
                        ),
                    },
                },
                "required": [
                    "train_journey",
                    "current_location",
                    "destination",
                    "current_delay",
                    "planned_time_at_current_stop",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_station_coverage",
            "description": (
                "Check whether the current location and destination are on the "
                "supported Weymouth ↔ Waterloo route before requesting a delay "
                "prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_location": {
                        "type": "string",
                        "description": "Station where the train currently is",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Passenger destination station",
                    },
                },
                "required": ["current_location", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_covered_stations",
            "description": (
                "Return the ordered list of stations covered for a journey "
                "direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "journey": {
                        "type": "string",
                        "description": "Journey direction: 'WEY2WAT' or 'WAT2WEY'",
                    }
                },
                "required": ["journey"],
            },
        },
    },
]
