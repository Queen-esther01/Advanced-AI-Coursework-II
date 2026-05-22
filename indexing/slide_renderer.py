import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _slide_sort_key(slide_xml_name: str) -> int:
    digits = "".join(filter(str.isdigit, slide_xml_name))
    return int(digits) if digits else 0


def extract_slide_texts(pptx_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with zipfile.ZipFile(pptx_path) as zf:
        slide_files = sorted(
            (
                f
                for f in zf.namelist()
                if f.startswith("ppt/slides/slide") and "_rels" not in f
            ),
            key=lambda s: _slide_sort_key(Path(s).name),
        )
        for slide_file in slide_files:
            root = ET.fromstring(zf.read(slide_file))
            runs = [
                t.text.strip()
                for t in root.findall(f".//{{{NS_A}}}t")
                if t.text and t.text.strip()
            ]
            texts[Path(slide_file).name] = "\n".join(runs)
    return texts


def _libreoffice_to_pdf(pptx_path: Path, work_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(
            "LibreOffice not found. Install it (e.g. brew install --cask libreoffice) "
            "to render CPT slides."
        )
    result = subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(work_dir),
            str(pptx_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice failed:\n{result.stderr or result.stdout}")
    pdf_path = work_dir / f"{pptx_path.stem}.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Expected PDF not found: {pdf_path}")
    return pdf_path


def _pdf_to_jpegs(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError(
            "pdftoppm not found. Install poppler (e.g. brew install poppler) to render CPT slides."
        )
    prefix = out_dir / "slide"
    result = subprocess.run(
        [pdftoppm, "-jpeg", "-r", str(dpi), str(pdf_path), str(prefix)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed:\n{result.stderr or result.stdout}")
    return sorted(out_dir.glob("slide-*.jpg"))


def render_slides(
    pptx_path: Path,
    out_dir: Path,
    *,
    dpi: int = 150,
    write_manifest: bool = True,
) -> list[dict]:
    pptx_path = Path(pptx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting slide text from {pptx_path.name}...")
    slide_texts = extract_slide_texts(pptx_path)
    sorted_keys = sorted(slide_texts.keys(), key=_slide_sort_key)

    print(f"  Rendering slides via LibreOffice ({dpi} dpi)...")
    pdf_path = _libreoffice_to_pdf(pptx_path, out_dir)
    jpeg_paths = _pdf_to_jpegs(pdf_path, out_dir, dpi=dpi)
    pdf_path.unlink(missing_ok=True)

    if len(jpeg_paths) != len(sorted_keys):
        print(
            f"  Warning: {len(jpeg_paths)} rendered pages vs {len(sorted_keys)} slide XML files",
            file=sys.stderr,
        )

    manifest: list[dict] = []
    for i, (jpeg, slide_key) in enumerate(
        zip(jpeg_paths, sorted_keys), start=1
    ):
        text = slide_texts.get(slide_key, "")
        txt_path = jpeg.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")
        title = _slide_title(text, i)
        manifest.append(
            {
                "slide_number": i,
                "slide_xml": slide_key,
                "title": title,
                "image_path": str(jpeg.resolve()),
                "text_path": str(txt_path.resolve()),
                "text": text,
            }
        )

    if write_manifest:
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"  Rendered {len(manifest)} slides → {out_dir}")
    return manifest


def _slide_title(text: str, slide_number: int) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return f"Slide {slide_number}"


def manifest_to_markdown(manifest: list[dict]) -> str:
    parts = []
    for entry in manifest:
        n = entry["slide_number"]
        title = entry.get("title") or f"Slide {n}"
        body = entry.get("text", "").strip()
        parts.append(f"## Slide {n}: {title}\n\n{body}")
    return "\n\n".join(parts)
