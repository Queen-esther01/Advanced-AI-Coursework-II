from dataclasses import dataclass
from pathlib import Path

import chromadb

from expert.station_index import metadata_station_values
from indexing.vector_store import VectorStore

_BOILERPLATE_SECTIONS = ("introduction", "top tips", "contents", "document control", "issue")
_OPERATIONAL_SECTIONS = (
    "communication",
    "control",
    "practical",
    "operation",
    "crowd",
    "welfare",
    "evacuation",
    "alternative",
    "bus",
    "taxi",
    "liaison",
    "mod ",
)
_ROLE_TERMS = {
    "signaller": ("signaller", "signalling", "signal"),
    "station_staff": ("station", "platform", "gateline", "dispatch"),
    "control": ("control", "controller", "operations"),
}
_TIME_TERMS = {
    "peak": ("peak", "rush", "commuter"),
    "off_peak": ("off peak", "off-peak", "overnight", "quiet period"),
}


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float | None = None


def build_where(plan_source: str, station_filter: str | None) -> dict | None:
    clauses: list[dict] = [{"document_type": {"$eq": plan_source}}]
    if station_filter:
        values = metadata_station_values(station_filter)
        if len(values) == 1:
            clauses.append({"station": {"$eq": values[0]}})
        else:
            clauses.append({"station": {"$in": values}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _section_rank_boost(section: str) -> float:
    s = section.lower()
    score = 0.0
    if any(term in s for term in _BOILERPLATE_SECTIONS):
        score -= 3.0
    if any(term in s for term in _OPERATIONAL_SECTIONS):
        score += 2.0
    return score


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> float:
    lower = text.lower()
    return float(sum(1 for keyword in keywords if keyword in lower))


def _context_rank_boost(
    chunk: RetrievedChunk,
    *,
    staff_role: str | None,
    service_period: str | None,
    derived_actions: list[str] | None,
) -> float:
    section = chunk.metadata.get("section", "")
    body = chunk.text
    score = _section_rank_boost(section)
    if staff_role and staff_role in _ROLE_TERMS:
        score += _keyword_hits(f"{section} {body}", _ROLE_TERMS[staff_role]) * 0.7
    if service_period and service_period in _TIME_TERMS:
        score += _keyword_hits(body, _TIME_TERMS[service_period]) * 0.6
    if derived_actions:
        action_terms = tuple(action.replace("_", " ") for action in derived_actions)
        score += _keyword_hits(body, action_terms) * 0.8
    return score


def _rerank_chunks(
    chunks: list[RetrievedChunk],
    *,
    limit: int,
    staff_role: str | None,
    service_period: str | None,
    derived_actions: list[str] | None,
) -> list[RetrievedChunk]:
    ranked = sorted(
        chunks,
        key=lambda c: (
            -_context_rank_boost(
                c,
                staff_role=staff_role,
                service_period=service_period,
                derived_actions=derived_actions,
            ),
            c.distance if c.distance is not None else 999.0,
        ),
    )
    return ranked[:limit]


def _metadata_fallback(
    vector_store: VectorStore,
    *,
    plan_source: str,
    station_filter: str | None,
    limit: int,
) -> list[RetrievedChunk]:
    where = build_where(plan_source, station_filter)
    if where is None:
        return []
    data = vector_store.collection.get(
        where=where,
        include=["documents", "metadatas"],
        limit=limit,
    )
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    return [
        RetrievedChunk(text=doc, metadata=meta, distance=None)
        for doc, meta in zip(docs, metas)
        if doc
    ]


def retrieve_plans(
    vector_store: VectorStore,
    *,
    query: str,
    plan_source: str,
    station_filter: str | None = None,
    staff_role: str | None = None,
    service_period: str | None = None,
    derived_actions: list[str] | None = None,
    n_results: int = 6,
) -> list[RetrievedChunk]:
    if vector_store.count() == 0:
        return []

    where = build_where(plan_source, station_filter)
    fetch_n = max(n_results * 3, 12)
    chunks: list[RetrievedChunk] = []

    try:
        results = vector_store.query(
            query_texts=[query],
            n_results=fetch_n,
            where=where,
        )
        chunks = _results_to_chunks(results)
    except chromadb.errors.InternalError as exc:
        if "finding id" not in str(exc).lower():
            raise
        chunks = _metadata_fallback(
            vector_store,
            plan_source=plan_source,
            station_filter=station_filter,
            limit=fetch_n,
        )

    return _rerank_chunks(
        chunks,
        limit=n_results,
        staff_role=staff_role,
        service_period=service_period,
        derived_actions=derived_actions,
    )


def _results_to_chunks(results: dict) -> list[RetrievedChunk]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] or [None] * len(docs)
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for doc, meta, dist in zip(docs, metas, dists):
        key = f"{meta.get('source')}|{meta.get('section')}|{meta.get('slide_number')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(RetrievedChunk(text=doc, metadata=meta, distance=dist))
    return out


def format_context(chunks: list[RetrievedChunk], max_chars: int = 12000) -> str:
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        header = (
            f"[Source {i}: {meta.get('source', '?')} | "
            f"{meta.get('section', '?')} | {meta.get('document_type', '?')}]"
        )
        block = f"{header}\n{chunk.text.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def image_paths_from_chunks(chunks: list[RetrievedChunk]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for chunk in chunks:
        meta = chunk.metadata
        for raw in (meta.get("image_path"), meta.get("source_image")):
            if not raw:
                continue
            path = Path(raw)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def source_labels(chunks: list[RetrievedChunk]) -> list[str]:
    labels = []
    for chunk in chunks:
        meta = chunk.metadata
        labels.append(
            f"{meta.get('source', 'plan')} — {meta.get('section', 'section')}"
        )
    return labels
