import hashlib
from pathlib import Path

import chromadb

from config import CHROMA_DB_PATH

DEFAULT_DB_PATH = CHROMA_DB_PATH
DEFAULT_COLLECTION_NAME = "disruption_plans"


def stable_chunk_id(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    payload = "|".join(
        [
            meta.get("source", ""),
            meta.get("station", ""),
            meta.get("section", ""),
            meta.get("type", "text"),
            str(meta.get("slide_number", "")),
            meta.get("source_image", ""),
            meta.get("filename", ""),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_chunk_ids(chunks: list[dict]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for i, chunk in enumerate(chunks):
        chunk_id = stable_chunk_id(chunk)
        if chunk_id in seen:
            chunk_id = hashlib.sha256(f"{chunk_id}|{i}".encode()).hexdigest()
        seen.add(chunk_id)
        ids.append(chunk_id)
    return ids


class VectorStore:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            print("VectorStore: no chunks to add")
            return 0
        image_count = sum(1 for c in chunks if c.get("metadata", {}).get("type") == "image")
        text_count = len(chunks) - image_count
        ids = stable_chunk_ids(chunks)
        print(
            f"VectorStore: upserting {len(chunks)} chunks "
            f"({text_count} text, {image_count} image) → '{self.collection_name}'"
        )
        self.collection.upsert(
            ids=ids,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        print(f"VectorStore: collection now has {self.count()} documents")
        return len(chunks)

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict = {"query_texts": query_texts, "n_results": n_results}
        if where is not None:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    @staticmethod
    def image_paths_from_results(results: dict) -> list[Path]:
        paths = []
        for meta in results.get("metadatas", [[]])[0]:
            if meta.get("type") == "image" and meta.get("image_path"):
                paths.append(Path(meta["image_path"]))
        return paths
