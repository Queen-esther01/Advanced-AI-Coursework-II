import pathlib
from pathlib import Path

from dotenv import load_dotenv
from IPython.display import Image, display

from config import EXTRACTED_MEDIA_DIR, STORE_DIR
from indexing.data_chunker import DataChunker
from indexing.vector_store import VectorStore
from llm_client import LLMClient

load_dotenv()


def main():
    data_path = pathlib.Path().cwd() / "data"
    disruption_plans_path = data_path / "disruption_plans"
    print(pathlib.Path().cwd())
    print(f"Store: {STORE_DIR.resolve()}")

    vector_store = VectorStore()

    print(f"Collection has {vector_store.count()} existing documents")

    llm_client = LLMClient()
    chunker = DataChunker(llm_client)

    first_data_path = (
        disruption_plans_path
        / "SWR Station Disruption Plans/RM Central/Station Disruption Plan - Aldershot Issue 1 - April 2018.docx"
    )

    print("--- Chunking ---")
    chunks = chunker.chunk_file(first_data_path)

    print("\n--- Indexing into Chroma ---")
    n = vector_store.add_chunks(chunks)

    print(f"\nSummary: {n} chunks indexed from {first_data_path.name}")
    print(f"Collection total: {vector_store.count()} documents")

    extracted_files = [p for p in EXTRACTED_MEDIA_DIR.rglob("*") if p.is_file()]
    print(f"Files in .store/extracted_media: {len(extracted_files)}")
    for p in sorted(extracted_files):
        print(f"  {p.relative_to(STORE_DIR)}")

    # chunked = chunker.chunk_directory(disruption_plans_path)
    # all_chunks = [chunk for chunks in chunked.values() for chunk in chunks]
    # n = vector_store.add_chunks(all_chunks)
    # print(f"\nIndexed {n} chunks across {len(chunked)} files")
    # print(f"Collection now has {vector_store.count()} documents")

    results = vector_store.query(
        query_texts=["Aldershot station incident"],
        n_results=5,
    )

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"--- Result {i + 1} [{meta.get('station')} / {meta.get('section')}] ---")
        print(doc[:400])
        image_path = meta.get("image_path")
        if image_path and Path(image_path).exists():
            print(f"Image: {image_path}")
            display(Image(filename=image_path))
        print()


if __name__ == "__main__":
    main()
