from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

STORE_DIR = PROJECT_ROOT / ".store"
CHROMA_DB_PATH = STORE_DIR / "chroma_db"
EXTRACTED_MEDIA_DIR = STORE_DIR / "extracted_media"
IMAGE_DESCRIPTIONS_CACHE_PATH = STORE_DIR / "image_descriptions_cache.json"

# Input files
INTENTIONS_PATH = DATA_DIR / "intentions.json"
SENTENCES_PATH = DATA_DIR / "sentences.txt"
STATIONS_CSV_PATH = DATA_DIR / "StationNameAndCode.csv"

# Global flag used to enable extra prompts in the final chatbot
final_chatbot = True