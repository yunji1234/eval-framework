"""
config.py — Shared configuration constants for the HR Policy RAG system.
"""

import os
from dotenv import load_dotenv

# Must run before reading env vars — this module is imported before the
# callers' load_dotenv() calls execute, so we load .env here directly.
load_dotenv()

COLLECTION_NAME  = "hr_policies"
DEFAULT_DB_PATH  = os.getenv("RAG_DB_PATH", "./chroma_db")
DEFAULT_DATA_DIR = "./documents"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
