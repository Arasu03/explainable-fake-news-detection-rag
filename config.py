from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# A 3-way NLI model: entailment, contradiction, neutral.
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

TOP_K = 4
ENTAIL_THRESHOLD = 0.70
CONTRADICT_THRESHOLD = 0.70
CHUNK_WORDS = 160
CHUNK_OVERLAP = 30

KB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
