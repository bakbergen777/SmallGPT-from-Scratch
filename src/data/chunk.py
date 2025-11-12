import argparse, json, re, pathlib, sys

def clean(txt:str)->str:
    txt = txt.replace('\r',' ')
    txt = re.sub(r'[ \t]+',' ',txt)
    txt = re.sub(r'\n{3,}','\n\n',txt)
    return txt.strip()

def make_chunks(txt, size=1000, overlap=200):
    assert size > 0, "chunk size must be > 0"
    assert 0 <= overlap < size, "0 <= overlap < chunk size"
    out=[]; i=0; n=len(txt); step = size - overlap
    while i < n:
        j = min(i + size, n)
        out.append({"text": txt[i:j], "start": i, "end": j})
        i += max(1, step)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    p_in = pathlib.Path(args.inp)
    if not p_in.exists():
        print(f"File not found: {p_in}", file=sys.stderr); sys.exit(1)

    text = p_in.read_text(encoding="utf-8", errors="ignore")
    text = clean(text)
    chunks = make_chunks(text, args.chunk, args.overlap)

    p_out = pathlib.Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with p_out.open("w", encoding="utf-8") as f:
        for i,c in enumerate(chunks):
            f.write(json.dumps({"id": i, **c})+"\n")
    print(f"✅ Saved {len(chunks)} chunks → {p_out}")
