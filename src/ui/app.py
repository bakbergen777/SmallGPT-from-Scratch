import json, subprocess, sys
from pathlib import Path
import torch, gradio as gr

from src.data.chunk import clean, make_chunks
from src.data.tfidf_index import build as build_index, search as search_index, load_index
from src.util.io import read_text
from src.util.tokenizer import CharTokenizer
from src.model.gpt_mini import MiniGPT

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
IDX  = DATA / "indices"
CKPT_DIR = ROOT / "checkpoints"
CKPT = CKPT_DIR / "minigpt.pt"
for p in [PROC, IDX, CKPT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def dev():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def _save_chunks(chunks, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"id": i, **c}, ensure_ascii=False)+"\n")

def _prompt(ctx_docs, q: str) -> str:
    body = "\n\n".join([d for _,_,d in ctx_docs])
    return f"Use the following context to answer.\n\n{body}\n\nQ: {q}\nA:"

# -------- Tab 1: Ask --------
def ui_load_text(file, index_state, tok_state):
    if file is None:
        return index_state, tok_state, "❌ Upload a .txt first."
    p = Path(file.name)
    text = read_text(p)
    text = clean(text)
    chunks = make_chunks(text, chunk_size=1000, overlap=200)
    out_jsonl = PROC / "chunks.jsonl"
    _save_chunks(chunks, out_jsonl)
    build_index(out_jsonl, IDX)
    index = load_index(IDX)
    tok = CharTokenizer(text)
    return index, tok, f"✅ {p.name} loaded | {len(chunks)} chunks | TF-IDF ready."

def ui_load_ckpt(index_state, tok_state, ckpt_path):
    if tok_state is None:
        return None, "❌ Load text first (tokenizer needed)."
    ckpt = Path(ckpt_path.strip()) if ckpt_path and ckpt_path.strip() else CKPT
    if not ckpt.exists():
        return None, f"❌ Checkpoint not found: {ckpt}"
    sd = torch.load(ckpt, map_location="cpu")
    cfg = sd.get("cfg", {})
    cfg = dict(
        ctx=cfg.get("ctx", 128),
        d_model=cfg.get("d_model", 128),
        n_layers=cfg.get("n_layers", cfg.get("layers", 2)),
        n_heads=cfg.get("n_heads",  cfg.get("heads", 4)),
    )
    model = MiniGPT(tok_state.vocab_size, **cfg).to(dev())
    model.load_state_dict(sd["model"])
    model.eval()
    return model, f"✅ Loaded {ckpt.name} {cfg}"

@torch.no_grad()
def ui_ask(index_state, tok_state, model_state, q, k, max_new, temperature, top_k):
    if index_state is None: return "❌ Index not ready."
    if tok_state is None:   return "❌ Tokenizer not ready."
    if model_state is None: return "❌ Model not loaded."
    k = max(1, int(k))
    top = search_index(index_state, q, k)
    prompt = _prompt(top, q)
    enc = torch.tensor([tok_state.encode(prompt)], dtype=torch.long, device=dev())
    temperature = max(float(temperature), 1e-6)
    tk = None
    if top_k is not None and int(top_k) > 0:
        tk = min(int(top_k), model_state.head.out_features)
    out = model_state.generate(enc, max_new=int(max_new), temperature=temperature, top_k=tk)
    ans = tok_state.decode(out[0].tolist()[len(tok_state.encode(prompt)):]).strip()
    return ans or "(no output)"

# -------- Tab 2: Train (subprocess, live logs) --------
def ui_train(data_path, steps, ctx, dmodel, n_layers, n_heads, bs):
    py = sys.executable
    cmd = [
        py, "-m", "src.train.train",
        "--data", data_path,
        "--steps", str(int(steps)),
        "--ctx", str(int(ctx)),
        "--dmodel", str(int(dmodel)),
        "--layers", str(int(n_layers)),
        "--heads", str(int(n_heads)),
        "--bs", str(int(bs)),
    ]
    yield "🚀 " + " ".join(cmd)
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            for line in p.stdout:
                yield line.rstrip("\n")
        code = p.wait()
        yield "\n✅ Training complete. Saved → checkpoints/minigpt.pt" if code == 0 else f"\n❌ Exit code {code}"
    except Exception as e:
        yield f"❌ {e!r}"

# -------- Theme + App --------
try:
    THEME = gr.themes.Soft(primary_hue="green")
except Exception:
    THEME = None
CSS = """
.gradio-container { max-width: 980px !important; }
body, .gradio-container { background: #0b0f0c !important; color: #e6f4ea !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="ADA — SmallGPT (Dark)") as demo:
    gr.Markdown("## 🟩 ADA — SmallGPT\nMinimal local RAG + tiny GPT")

    with gr.Tabs():
        with gr.TabItem("Ask"):
            with gr.Row():
                with gr.Column():
                    f_in = gr.File(label="Upload .txt")
                    b_load = gr.Button("Load & Build Index", variant="primary")
                    status = gr.Markdown()
                    ckpt_in = gr.Textbox(value=str(CKPT), label="Checkpoint path")
                    b_ckpt = gr.Button("Load checkpoint")
                with gr.Column():
                    q = gr.Textbox(label="Question", placeholder="What is Chapter 2 about?")
                    k = gr.Slider(1, 10, value=3, step=1, label="Top-K passages")
                    max_new = gr.Slider(32, 512, value=150, step=1, label="Max new tokens")
                    temperature = gr.Slider(0.2, 1.5, value=0.9, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 200, value=50, step=1, label="Sampling Top-K (0=off)")
                    b_ask = gr.Button("Ask", variant="primary")
                    out = gr.Textbox(label="Answer", lines=12)
            index_state = gr.State(None)
            tok_state = gr.State(None)
            model_state = gr.State(None)
            b_load.click(ui_load_text, inputs=[f_in, index_state, tok_state], outputs=[index_state, tok_state, status])
            b_ckpt.click(ui_load_ckpt, inputs=[index_state, tok_state, ckpt_in], outputs=[model_state, status])
            b_ask.click(ui_ask, inputs=[index_state, tok_state, model_state, q, k, max_new, temperature, top_k], outputs=out)

        with gr.TabItem("Train tiny model"):
            gr.Markdown("Runs your existing trainer in a subprocess and streams logs.")
            data_path = gr.Textbox(value=str(RAW / "book.txt"), label="Data path (.txt)")
            steps = gr.Slider(50, 2000, value=300, step=10, label="Steps")
            ctx = gr.Slider(32, 512, value=128, step=16, label="Context")
            dmodel = gr.Slider(64, 512, value=128, step=16, label="d_model")
            n_layers = gr.Slider(1, 8, value=2, step=1, label="Layers")
            n_heads  = gr.Slider(1, 8, value=4, step=1, label="Heads")
            bs = gr.Slider(4, 64, value=16, step=1, label="Batch size")
            b_train = gr.Button("Start training", variant="primary")
            log = gr.Textbox(label="Training log", lines=18)
            b_train.click(ui_train, inputs=[data_path, steps, ctx, dmodel, n_layers, n_heads, bs], outputs=log)

if __name__ == "__main__":
    demo.launch()
