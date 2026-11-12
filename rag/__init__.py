"""A from-scratch RAG pipeline: load -> chunk -> embed -> store -> retrieve -> generate."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

load_dotenv(ROOT / ".env")
