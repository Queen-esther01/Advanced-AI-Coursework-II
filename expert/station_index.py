import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_STATIONS_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "disruption_plans" / "stations.csv"
)


@dataclass(frozen=True)
class _StationAliases:
    canonical: str
    aliases: tuple[str, ...]


def _clean(value: str) -> str:
    return " ".join(value.strip().split())


def _title_name(value: str) -> str:
    lowered = value.lower()
    return lowered.title() if lowered.isupper() else value


def _strip_suffix(value: str) -> str:
    return re.sub(r"\s+rail station$", "", value, flags=re.I).strip()


def _variants(value: str) -> set[str]:
    v = _clean(value)
    out = {v}
    stripped = _strip_suffix(v)
    out.add(stripped)
    if stripped.lower().startswith("london "):
        out.add(stripped[7:])
        out.add(f"{stripped[7:]} London")
    return {item for item in out if item}


@lru_cache(maxsize=1)
def _station_maps() -> tuple[dict[str, _StationAliases], dict[str, tuple[str, ...]]]:
    alias_lookup: dict[str, _StationAliases] = {}
    metadata_lookup: dict[str, tuple[str, ...]] = {}

    with _STATIONS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name_raw = (row.get("name") or "").strip()
            if not name_raw:
                continue

            canonical = _title_name(_strip_suffix(name_raw))
            alias_set: set[str] = set()
            for field in ("name", "longname.name_alias"):
                field_value = (row.get(field) or "").strip()
                if not field_value or field_value == r"\N":
                    continue
                alias_set.update(_variants(_title_name(field_value)))

            for field in ("alpha3", "tiploc"):
                field_value = (row.get(field) or "").strip()
                if field_value and field_value != r"\N":
                    alias_set.add(field_value.upper())

            aliases = tuple(sorted(alias_set, key=lambda item: (len(item), item.lower())))
            station_aliases = _StationAliases(canonical=canonical, aliases=aliases)
            metadata_lookup[canonical.lower()] = aliases
            for alias in aliases:
                alias_lookup[alias.lower()] = station_aliases

    return alias_lookup, metadata_lookup


INDEXED_STATION_HINT = "indexed station plans (use full name or CRS code)"


def canonical_station_name(name: str) -> str:
    alias_lookup, _ = _station_maps()
    key = _clean(name).lower()
    if not key:
        return name
    return alias_lookup.get(key, _StationAliases(name, (name,))).canonical


def metadata_station_values(canonical_or_alias: str) -> list[str]:
    _, metadata_lookup = _station_maps()
    key = canonical_station_name(canonical_or_alias).lower()
    aliases = metadata_lookup.get(key)
    if aliases:
        return list(aliases)
    raw = _clean(canonical_or_alias)
    return [raw] if raw else []
