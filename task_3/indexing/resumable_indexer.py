import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from config import STORE_DIR
from indexing.data_chunker import DataChunker
from indexing.vector_store import VectorStore

DEFAULT_DOCUMENT_SUFFIXES = (".doc", ".docx", ".docm", ".pptx", ".pptm")
DEFAULT_CHECKPOINT_PATH = STORE_DIR / "indexing_checkpoint.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def discover_documents(
    root: Path, *, suffixes: tuple[str, ...] = DEFAULT_DOCUMENT_SUFFIXES
) -> list[Path]:
    root = Path(root).resolve()
    allowed = {s.lower() for s in suffixes}
    files = [p.resolve() for p in root.rglob("*") if p.is_file()]
    return sorted(p for p in files if p.suffix.lower() in allowed)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)

class IndexingCheckpoint:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH):
        self.path = Path(checkpoint_path)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "updated_at": _utc_now(), "files": {}}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
            parsed.setdefault("version", 1)
            parsed.setdefault("updated_at", _utc_now())
            parsed.setdefault("files", {})
            return parsed
        except Exception:
            return {"version": 1, "updated_at": _utc_now(), "files": {}}

    def save(self) -> None:
        self.data["updated_at"] = _utc_now()
        _atomic_write_json(self.path, self.data)

    def status(self, path: Path) -> str | None:
        return self.data["files"].get(str(path.resolve()), {}).get("status")

    def mark(self, path: Path, *, status: str, chunk_count: int = 0, error: str = "") -> None:
        self.data["files"][str(path.resolve())] = {
            "status": status,
            "chunk_count": chunk_count,
            "error": error,
            "updated_at": _utc_now(),
        }
        self.save()


def _chunk_one(path: Path, chunker: DataChunker) -> tuple[Path, list[dict]]:
    chunks = chunker.chunk_file(path)
    return path, chunks


def run_resumable_indexing(
    *,
    paths: list[Path],
    chunker: DataChunker,
    vector_store: VectorStore,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    max_workers: int = 4,
    retry_failed: bool = False,
) -> dict[str, int]:
    checkpoint = IndexingCheckpoint(checkpoint_path)

    normalized = [Path(p).resolve() for p in paths]
    pending: list[Path] = []
    for p in normalized:
        status = checkpoint.status(p)
        if status == "done":
            continue
        if status == "failed" and not retry_failed:
            continue
        pending.append(p)

    for p in pending:
        checkpoint.mark(p, status="in_progress")

    summary = {
        "total_input": len(normalized),
        "pending": len(pending),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "chunks_upserted": 0,
    }
    if not pending:
        print("No pending files to process (all done or failed with retry_failed=False).")
        return summary

    workers = max(1, int(max_workers))
    print(f"Starting resumable indexing with {workers} worker(s) over {len(pending)} file(s).")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_chunk_one, path, chunker): path for path in pending}
        for i, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            summary["processed"] += 1
            try:
                _path, chunks = future.result()
                n = vector_store.add_chunks(chunks)
                checkpoint.mark(path, status="done", chunk_count=n)
                summary["succeeded"] += 1
                summary["chunks_upserted"] += n
                print(f"[{i}/{len(pending)}] DONE {path.name} ({n} chunks)")
            except Exception as exc:
                checkpoint.mark(path, status="failed", error=str(exc))
                summary["failed"] += 1
                print(f"[{i}/{len(pending)}] FAIL {path.name}: {exc}")

    return summary
