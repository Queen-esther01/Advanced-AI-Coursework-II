import hashlib
import json
import re
import shutil
from pathlib import Path

from config import EXTRACTED_MEDIA_DIR, IMAGE_DESCRIPTIONS_CACHE_PATH, STORE_DIR
from llm_client import LLMClient

from .doc_parser import (
    chunk_markdown,
    parse_image_markdown,
    read_docx,
    read_pptx_slides,
)
from .image_prompts import document_kind, image_description_prompt, slide_guide_prompt

DEFAULT_DOCUMENT_GLOBS = ("**/*.docx", "**/*.docm", "**/*.pptm", "**/*.pptx")

DEFAULT_CACHE_PATH = IMAGE_DESCRIPTIONS_CACHE_PATH
DEFAULT_EXTRACTED_MEDIA_DIR = EXTRACTED_MEDIA_DIR


def _station_name_from_path(path: Path) -> str:
    stem = path.stem
    match = re.search(r"Plan - (.+?) Issue", stem)
    if match:
        return match.group(1)
    match = re.search(r"^(.+?)\s+CPT\b", stem, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stem


def _station_slug(station: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", station).strip().lower()
    return re.sub(r"[-\s]+", "-", slug) or "unknown"


def _safe_filename(name: str) -> str:
    name = Path(name).name
    safe = "".join(c for c in name if c.isalnum() or c in ".-_")
    return safe or "image.bin"


def _format_image_description(result: dict[str, str]) -> str:
    return f"[Image: {result['suggested_filename']}] {result['caption']} {result['description']}".strip()


def _format_slide_guide(section: str, result: dict[str, str]) -> str:
    guide = (result.get("guide") or "").strip()
    if not guide:
        return _format_image_description(result)
    return f"{section}\n\n{guide}".strip()


def chunk_length_stats(chunks: list[dict]) -> dict:
    lengths = [len(c.get("text", "")) for c in chunks]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def print_chunk_length_stats(chunks: list[dict], label: str = "Chunks") -> dict:
    stats = chunk_length_stats(chunks)
    print(
        f"{label}: {stats['count']} chunks | "
        f"min={stats['min']} max={stats['max']} mean={stats['mean']:.0f} chars"
    )
    text_lengths = [
        len(c["text"]) for c in chunks if c.get("metadata", {}).get("type") != "image"
    ]
    image_lengths = [
        len(c["text"]) for c in chunks if c.get("metadata", {}).get("type") == "image"
    ]
    if text_lengths:
        print(
            f"  text:  min={min(text_lengths)} max={max(text_lengths)} "
            f"mean={sum(text_lengths) / len(text_lengths):.0f}"
        )
    if image_lengths:
        print(
            f"  image: min={min(image_lengths)} max={max(image_lengths)} "
            f"mean={sum(image_lengths) / len(image_lengths):.0f}"
        )
    return stats


class ImageDescriptionCache:
    def __init__(self, cache_path: Path = DEFAULT_CACHE_PATH):
        self.cache_path = Path(cache_path)
        self._entries: dict[str, dict] = {}
        self._load()

    def _cache_key(self, image_path: Path) -> str:
        return hashlib.sha256(str(image_path.resolve()).encode()).hexdigest()

    def _entry_from_raw(self, entry: dict) -> dict:
        return dict(entry)

    def _dedupe_by_source(self, entries: dict[str, dict]) -> dict[str, dict]:
        by_source: dict[str, tuple[str, dict]] = {}
        for key, entry in entries.items():
            source = entry.get("source_path", "")
            if not source:
                continue
            resolved = str(Path(source).resolve())
            existing = by_source.get(resolved)
            if existing is None:
                by_source[resolved] = (key, entry)
                continue
            _, prev = existing
            prev_path = (prev.get("extracted_path") or "").strip()
            new_path = (entry.get("extracted_path") or "").strip()
            if prev_path and not Path(prev_path).is_file() and new_path:
                by_source[resolved] = (key, entry)
            elif not prev_path and new_path:
                by_source[resolved] = (key, entry)

        deduped: dict[str, dict] = {}
        for _source, (_old_key, entry) in by_source.items():
            src = Path(entry.get("source_path", ""))
            if src.is_file():
                deduped[self._cache_key(src)] = entry
            elif entry.get("source_path"):
                deduped[self._cache_key(Path(entry["source_path"]))] = entry
        return deduped

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            raw = data.get("entries", {})
            deduped = self._dedupe_by_source(raw)
            self._entries = deduped
            if len(deduped) != len(raw):
                self._save()
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"version": 2, "entries": self._entries}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    def get(self, image_path: Path) -> dict[str, str] | None:
        key = self._cache_key(image_path)
        entry = self._entries.get(key)
        if entry:
            return self._entry_from_raw(entry)
        source = str(image_path.resolve())
        for cached in self._entries.values():
            if cached.get("source_path") == source:
                return self._entry_from_raw(cached)
        return None

    def set(self, image_path: Path, result: dict[str, str]) -> None:
        source = str(image_path.resolve())
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.get("source_path", "") != source
        }
        key = self._cache_key(image_path)
        entry = {k: v for k, v in result.items()}
        entry["source_path"] = source
        entry.setdefault("section", "")
        entry.setdefault("extracted_path", "")
        section = entry.get("section", "")
        if entry.get("guide"):
            entry["text"] = _format_slide_guide(section, entry)
        else:
            entry["text"] = _format_image_description(result)
        self._entries[key] = entry
        self._save()


class DataChunker:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        extracted_media_dir: Path = DEFAULT_EXTRACTED_MEDIA_DIR,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
        use_cache: bool = True,
    ):
        self.llm_client = llm_client
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.extracted_media_dir = Path(extracted_media_dir).resolve()
        self.use_cache = use_cache
        self.image_cache = (
            ImageDescriptionCache(cache_path) if use_cache and cache_path else None
        )
        print(f"DataChunker ready (store: {STORE_DIR.resolve()})")
        print(f"  extracted_media: {self.extracted_media_dir}")
        if self.image_cache:
            print(f"  image cache: {self.image_cache.cache_path}")

    def _expected_extracted_path(self, source: Path, station: str) -> Path:
        return (
            self.extracted_media_dir
            / _station_slug(station)
            / _safe_filename(source.name)
        )

    def _save_extracted_image(self, source: Path, station: str) -> Path:
        dest = self._expected_extracted_path(source, station)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return dest.resolve()

    def _extract_image(self, source: Path, station: str) -> Path:
        dest = self._expected_extracted_path(source, station)
        if dest.is_file():
            return dest.resolve()
        return self._save_extracted_image(source, station)

    def _describe_image(
        self,
        image_path: Path,
        section_title: str,
        alt: str,
        station: str,
        doc_kind: str,
        *,
        slide_text: str | None = None,
        slide_number: int | None = None,
    ) -> tuple[str, dict]:
        if slide_number is not None:
            extracted = image_path.resolve()
            cache_path = extracted
        else:
            extracted = self._extract_image(image_path, station)
            cache_path = image_path

        cached = self.image_cache.get(cache_path) if self.image_cache else None
        if cached:
            print(f"  Cache hit: {image_path.name} ({section_title})")
            result = cached
        else:
            print(f"  LLM describing: {image_path.name} ({section_title})")
            result = self.llm_client.describe_image(
                extracted,
                prompt=image_description_prompt(doc_kind),
                section_title=section_title,
                original_filename=alt,
                slide_text=slide_text,
            )
            result["extracted_path"] = str(extracted)
            result["source_path"] = str(cache_path.resolve())
            result["section"] = section_title
            if self.image_cache:
                self.image_cache.set(cache_path, result)

        text = _format_image_description(result)

        _SCALAR_TYPES = (str, int, float, bool)
        metadata = {
            "type": "image",
            "station": station,
            "section": section_title,
            "caption": result["caption"],
            "image_path": str(extracted),
            "filename": result["suggested_filename"],
            "source_image": str(cache_path.resolve()),
        }
        if slide_number is not None:
            metadata["slide_number"] = slide_number
        _skip = {
            "caption",
            "description",
            "suggested_filename",
            "section",
            "source_path",
            "extracted_path",
            "text",
        }
        for key, value in result.items():
            if key in _skip or key in metadata:
                continue
            if isinstance(value, list):
                metadata[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, _SCALAR_TYPES):
                metadata[key] = value
            elif value is not None:
                metadata[key] = str(value)

        return text, metadata

    def _synthesize_slide_guide(
        self,
        section: str,
        slide_text: str,
        vision_result: dict,
        doc_kind: str,
        cache_path: Path,
    ) -> dict:
        if "guide" in vision_result:
            return vision_result

        print(f"  LLM guide: {section}")
        guide_result = self.llm_client.generate_slide_guide(
            prompt=slide_guide_prompt(doc_kind),
            section_title=section,
            slide_text=slide_text,
            image_description=vision_result,
        )
        merged = {**vision_result, **guide_result}
        if self.image_cache:
            self.image_cache.set(cache_path, merged)
        return merged

    def _describe_cpt_slide(
        self,
        entry: dict,
        station: str,
        doc_kind: str,
    ) -> dict | None:
        image_path = Path(entry["image_path"])
        if not image_path.is_file():
            print(f"  Skipping missing slide render: {image_path}")
            return None

        n = entry["slide_number"]
        title = entry.get("title") or f"Slide {n}"
        section = f"Slide {n}: {title}"
        slide_text = entry.get("text", "")
        cache_path = image_path.resolve()

        cached = self.image_cache.get(cache_path) if self.image_cache else None
        if cached:
            print(f"  Cache hit: {image_path.name} ({section})")
            vision_result = cached
        else:
            print(f"  LLM describing: {image_path.name} ({section})")
            vision_result = self.llm_client.describe_image(
                cache_path,
                prompt=image_description_prompt(doc_kind),
                section_title=section,
                original_filename=image_path.name,
                slide_text=slide_text,
            )
            vision_result["extracted_path"] = str(cache_path)
            vision_result["source_path"] = str(cache_path)
            vision_result["section"] = section
            if self.image_cache:
                self.image_cache.set(cache_path, vision_result)

        result = self._synthesize_slide_guide(
            section, slide_text, vision_result, doc_kind, cache_path
        )
        if not (result.get("guide") or "").strip():
            print(f"  Skip index/menu slide (empty guide): {section}")
            return None

        chunk_text = _format_slide_guide(section, result)

        metadata = {
            "type": "slide_guide",
            "station": station,
            "section": section,
            "caption": result.get("caption", ""),
            "image_path": str(cache_path),
            "filename": result.get("suggested_filename", ""),
            "source_image": str(cache_path),
            "slide_number": n,
        }
        _skip = {
            "caption",
            "description",
            "guide",
            "suggested_filename",
            "section",
            "source_path",
            "extracted_path",
            "text",
        }
        _SCALAR_TYPES = (str, int, float, bool)
        for key, value in result.items():
            if key in _skip or key in metadata:
                continue
            if isinstance(value, list):
                metadata[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, _SCALAR_TYPES):
                metadata[key] = value
            elif value is not None:
                metadata[key] = str(value)

        return {"text": chunk_text, "metadata": metadata}

    def _describe_slides(
        self, manifest: list[dict], station: str, doc_kind: str
    ) -> list[dict]:
        chunks = []
        for entry in manifest:
            chunk = self._describe_cpt_slide(entry, station, doc_kind)
            if chunk:
                chunks.append(chunk)
        print(f"  Built {len(chunks)} slide guide(s)")
        return chunks

    def _substitute_images(
        self, content: str, station: str, doc_kind: str
    ) -> tuple[str, list[dict]]:
        lines = content.splitlines()
        current_section = ""
        out = []
        image_chunks = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                current_section = stripped.lstrip("# ").strip()
            elif stripped.startswith("## Slide "):
                current_section = stripped.lstrip("# ").strip()
            parsed = parse_image_markdown(stripped)
            if parsed:
                alt, image_path = parsed
                if not image_path.exists():
                    print(f"  Skipping missing image: {image_path}")
                    continue
                text, img_meta = self._describe_image(
                    image_path, current_section, alt, station, doc_kind
                )
                out.append(text)
                image_chunks.append({"text": text, "metadata": img_meta})
                continue
            out.append(line)
        print(f"  Found {len(image_chunks)} image(s)")
        return "\n".join(out), image_chunks

    def _read_document(
        self, path: Path, station: str
    ) -> tuple[str, str, list[dict] | None]:
        kind = document_kind(path)
        if kind == "cpt_presentation":
            slides_dir = self.extracted_media_dir / _station_slug(station) / "slides"
            content, manifest = read_pptx_slides(path, slides_dir)
            return content, kind, manifest
        return read_docx(path), kind, None

    def chunk_file(self, path: Path, metadata: dict | None = None) -> list[dict]:
        path = Path(path)
        if metadata is None:
            metadata = {
                "source": path.name,
                "station": _station_name_from_path(path),
            }
        station = metadata["station"]
        print(f"Chunking: {path.name}")
        print(f"  Station: {station}")

        print("  Step 1/3: Reading document...")
        content, doc_kind, slide_manifest = self._read_document(path, station)
        metadata["document_type"] = doc_kind

        print("  Step 2/3: Describing slides / images (cache / LLM)...")
        if slide_manifest is not None:
            image_chunks = self._describe_slides(slide_manifest, station, doc_kind)
            text_chunks = []
            print("  Step 3/3: Slide guides only (no separate raw-text chunks)")
        else:
            content, image_chunks = self._substitute_images(content, station, doc_kind)
            print("  Step 3/3: Splitting into section chunks...")
            text_chunks = chunk_markdown(content, metadata)

        all_chunks = text_chunks + image_chunks
        for chunk in all_chunks:
            chunk["metadata"] = {**metadata, **chunk.get("metadata", {})}

        print(
            f"  Done: {len(text_chunks)} text + {len(image_chunks)} image/guide "
            f"= {len(all_chunks)} total"
        )
        print_chunk_length_stats(all_chunks, label=f"  {path.name}")
        return all_chunks

    def chunk_directory(
        self,
        directory: Path,
        glob: str | tuple[str, ...] | None = None,
    ) -> dict[str, list[dict]]:
        patterns = (
            (glob,) if isinstance(glob, str) else (glob or DEFAULT_DOCUMENT_GLOBS)
        )
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(directory.glob(pattern))
        results: dict[str, list[dict]] = {}
        for path in sorted({p.resolve() for p in paths}):
            try:
                chunks = self.chunk_file(path)
                results[path.name] = chunks
                print(f"  {path.name}: {len(chunks)} chunks")
            except Exception as exc:
                results[path.name] = []
                print(f"  {path.name}: ERROR — {exc}")
        all_chunks = [c for file_chunks in results.values() for c in file_chunks]
        if all_chunks:
            print()
            print_chunk_length_stats(all_chunks, label="Directory total")
        return results
