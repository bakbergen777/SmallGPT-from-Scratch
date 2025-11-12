import argparse, joblib, torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from src.model.gpt_mini import MiniGPT
from src.util.tokenizer import CharTokenizer

def dev():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def load_index(index_dir):
    return joblib.load(Path(index_dir)/"tfidf.joblib")

def search(index, query, k):
    vec, X, texts, ids = index["vec"], index["X"], index["texts"], index["ids"]
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top = sims.argsort()[::-1][:k]
    return [(ids[i], texts[i]) for i in top]

def build_prompt(chunks, q, max_chars=1500):
    ctx = ""
    for _,t in chunks:
        if len(ctx)+len(t)+2 > max_chars: break
        ctx += t.strip()+"\n\n"
    return f"System: You are a helpful book mentor. Answer only using the Context.\nContext:\n{ctx}\nQuestion: {q}\nAnswer:"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--index", default="data/indices")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ckpt", default="checkpoints/minigpt.pt")
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()

    book_text = Path("data/raw/book.txt").read_text(encoding="utf-8", errors="ignore")
    tok = CharTokenizer(book_text)

    # попытка загрузить чекпойнт (если нет — с нуля)
    cfg = dict(ctx=256, d_model=256, layers=4, heads=4)
    model = MiniGPT(tok.vocab_size, **cfg).to(dev())
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        model = MiniGPT(tok.vocab_size, **sd["cfg"]).to(dev())
        model.load_state_dict(sd["model"])
    model.eval()

    index = load_index(args.index)
    top = search(index, args.q, args.k)
    prompt = build_prompt(top, args.q)

    enc = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev())
    out = model.generate(enc, max_new=args.max_new, temperature=0.9, top_k=50)
    ans = tok.decode(out[0].tolist()[len(tok.encode(prompt)):])

    print("=== QUESTION ===")
    print(args.q)
    print("\n=== ANSWER ===")
    print(ans.strip())
