"""
Rebuild journey_metrics_lookup.csv from train_historical_data/*.xlsx.

Run from the repo root:
    python task_2/build_journey_lookup.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "train_historical_data"
OUT = DATA_DIR / "journey_metrics_lookup.csv"


def time_to_minutes(value):
    if pd.isna(value):
        return np.nan
    return value.hour * 60 + value.minute


def add_rollover_minutes(group, time_column):
    minutes = group[time_column].apply(time_to_minutes)
    rollover_days = 0
    previous_minute = np.nan
    absolute_minutes = []

    for minute in minutes:
        if pd.isna(minute):
            absolute_minutes.append(np.nan)
            continue

        if not pd.isna(previous_minute) and minute < previous_minute - (12 * 60):
            rollover_days += 1

        absolute_minutes.append(minute + (rollover_days * 24 * 60))
        previous_minute = minute

    return pd.Series(absolute_minutes, index=group.index)


def preprocess_journey_file(path):
    dataset = pd.read_excel(path)
    dataset.columns = dataset.columns.str.strip()
    dataset["location"] = dataset["location"].astype(str).str.strip().str.upper()
    dataset["_row_order"] = np.arange(len(dataset))
    dataset["rid"] = path.stem + "_" + dataset["rid"].astype(str)

    dataset = dataset.sort_values(["rid", "_row_order"]).copy()
    dataset["stop_number"] = dataset.groupby("rid").cumcount()
    for col, src in [
        ("planned_arrival_minutes", "planned_arrival_time"),
        ("planned_departure_minutes", "planned_departure_time"),
    ]:
        dataset[col] = dataset.groupby("rid", group_keys=False).apply(
            lambda group: add_rollover_minutes(group, src),
            include_groups=False,
        )

    dataset["current_planned_minutes"] = dataset["planned_arrival_minutes"].combine_first(
        dataset["planned_departure_minutes"]
    )
    return dataset


def build_lookup():
    records = []
    for path in sorted(DATA_DIR.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue

        stem = path.stem.upper()
        if stem.endswith("WEY2WAT"):
            journey = "WEY2WAT"
        elif stem.endswith("WAT2WEY"):
            journey = "WAT2WEY"
        else:
            continue

        print(f"Processing {path.name}...")
        dataset = preprocess_journey_file(path)
        for _, group in dataset.groupby("rid"):
            group = group.reset_index(drop=True)
            for i in range(len(group)):
                current_planned = group.at[i, "current_planned_minutes"]
                if pd.isna(current_planned):
                    continue

                current_station = group.at[i, "location"]
                dest_planned = group["planned_arrival_minutes"].iloc[i + 1 :].to_numpy()
                dest_stations = group["location"].iloc[i + 1 :].to_list()

                for offset, (dest_station, dest_plan) in enumerate(
                    zip(dest_stations, dest_planned), start=1
                ):
                    if pd.isna(dest_plan):
                        continue

                    remaining = dest_plan - current_planned
                    if remaining <= 0:
                        continue

                    records.append(
                        {
                            "journey": journey,
                            "current_station": current_station,
                            "destination_station": dest_station,
                            "stops_remaining": offset,
                            "remaining_minutes": remaining,
                        }
                    )

    raw = pd.DataFrame(records)
    lookup = raw.groupby(["journey", "current_station", "destination_station"]).agg(
        stops_remaining=("stops_remaining", "median"),
        remaining_minutes=("remaining_minutes", "median"),
        sample_count=("stops_remaining", "size"),
    ).reset_index()
    lookup["stops_remaining"] = lookup["stops_remaining"].round().astype(int)
    lookup["remaining_minutes"] = lookup["remaining_minutes"].round(1)
    lookup.to_csv(OUT, index=False)
    print(f"Saved {len(lookup):,} rows to {OUT}")


if __name__ == "__main__":
    build_lookup()
