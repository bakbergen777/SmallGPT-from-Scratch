import json
from pathlib import Path
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_SAVE = "tfidf.joblib"

def _read_chunks(chunks_path: Path) -> List[str]:
    texts = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
            except Exception:
                continue
    return texts

def build(chunks_path: Path, save_dir: Path) -> None:
    """Build TF-IDF index from JSONL chunks and save to save_dir/_SAVE."""
    chunks_path = Path(chunks_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    docs = _read_chunks(chunks_path)
    if not docs:
        raise ValueError(f"No docs found in {chunks_path}")

    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(docs)
    joblib.dump({"vec": vec, "X": X, "docs": docs}, save_dir / _SAVE)
    print(f"✅ TF-IDF index saved → {save_dir / _SAVE} ({len(docs)} docs, {len(vec.get_feature_names_out())} terms)")

def load_index(save_dir: Path):
    """Load TF-IDF index saved by build(). Returns a dict with vec, X, docs."""
    save_dir = Path(save_dir)
    path = save_dir / _SAVE
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    return joblib.load(path)

def search(index, query: str, topk: int = 5) -> List[Tuple[int, float, str]]:
    """Return topk (doc_id, score, text) by cosine similarity."""
    vec = index["vec"]
    X = index["X"]
    docs = index["docs"]
    qv = vec.transform([query])
    scores = linear_kernel(qv, X).ravel()
    topk = max(1, min(topk, len(docs)))
    ids = scores.argsort()[::-1][:topk]
    return [(int(i), float(scores[i]), docs[i]) for i in ids]
