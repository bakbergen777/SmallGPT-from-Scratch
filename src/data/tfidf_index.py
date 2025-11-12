import argparse, json, joblib, pathlib, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_chunks(path):
    texts, ids = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            ids.append(obj["id"]); texts.append(obj["text"])
    if not texts: raise ValueError("No chunks loaded.")
    return ids, texts

def build(chunks_path, save_dir, max_features=50000):
    ids, texts = load_chunks(chunks_path)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ids":ids,"texts":texts,"vec":vec,"X":X}, save_dir/"tfidf.joblib")
    print(f"✅ TF-IDF index saved → {save_dir/'tfidf.joblib'} ({X.shape[0]} docs, {X.shape[1]} terms)")

def search(load_dir, query, k=5):
    obj = joblib.load(pathlib.Path(load_dir)/"tfidf.joblib")
    vec, X, texts, ids = obj["vec"], obj["X"], obj["texts"], obj["ids"]
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top = sims.argsort()[::-1][:k]
    return [(int(ids[i]), float(sims[i]), texts[i]) for i in top]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--query")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.chunks and args.save:
        build(pathlib.Path(args.chunks), pathlib.Path(args.save))
        sys.exit(0)
    if args.load and args.query:
        for i,score,txt in search(args.load, args.query, args.topk):
            prev = txt[:300].replace("\n"," ")
            print(f"[{i}] score={score:.3f}\n{prev}\n---")
        sys.exit(0)
    print("Usage:\n  Build:  --chunks FILE --save DIR\n  Search: --load DIR --query 'text' [--topk K]")
    sys.exit(2)
