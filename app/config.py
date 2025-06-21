from dotenv import load_dotenv
import os

load_dotenv()  # Load .env into environment

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
QDRANT_URL = os.getenv("QDRANT_URL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multi_pdf_rag")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")