# ADA — SmallGPT

> A GPT-style transformer built from scratch in ~150 lines of PyTorch, combined with TF-IDF retrieval to answer questions about any book you upload — with a Gradio web UI for live training and inference.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio)
![MPS](https://img.shields.io/badge/Accelerator-MPS%20%2F%20CUDA%20%2F%20CPU-lightgrey)

---

## What It Does

ADA is a minimal **Retrieval-Augmented Generation (RAG)** system you can run entirely locally:

1. **Upload any `.txt` book** → it's chunked into 1 000-character passages and indexed with TF-IDF
2. **Train a tiny GPT** on that text (character-level, ~3 M parameters) directly from the UI
3. **Ask a question** → TF-IDF retrieves the most relevant passages → GPT generates an answer grounded in that context

Everything — training, retrieval, inference — runs in one Gradio browser tab with no external API.

---

## Demo

```
Question: What is the main theme of Chapter 2?

=== ANSWER ===
The second chapter explores the tension between individual will and collective duty,
drawing on the opening argument that no man can serve two masters without...
```

---

## Architecture

```
.txt file
    │
    ├─▶ CharTokenizer           (character-level vocab)
    │
    ├─▶ Chunker (1000 chars, 200 overlap)
    │       │
    │       └─▶ TF-IDF index (scikit-learn TfidfVectorizer + cosine similarity)
    │
    └─▶ MiniGPT training loop (AdamW, gradient clipping, MPS/CUDA/CPU)
            │
            └─▶ checkpoint saved → checkpoints/minigpt.pt

Query time:
    question ──▶ TF-IDF search (top-k passages)
                     │
                     └─▶ prompt = "Context:\n{passages}\nQ: {question}\nA:"
                                 │
                                 └─▶ MiniGPT.generate(temperature=0.9, top_k=50)
                                             │
                                             └─▶ answer text
```

---

## Model: MiniGPT

```python
MiniGPT(vocab, ctx=256, d_model=256, n_layers=4, n_heads=4)
```

| Component | Details |
|---|---|
| Embedding | token + learned positional |
| Attention | multi-head causal self-attention (masked upper triangle) |
| Block | Pre-LN: `x += Attn(LayerNorm(x))`, `x += FFN(LayerNorm(x))` |
| FFN | Linear → GELU → Linear (4× expansion) |
| Output | weight-tied linear head (shares weights with token embedding) |
| Context | 256 tokens (configurable) |
| Parameters | ~3 M at default settings |

The full model is in [`src/model/gpt_mini.py`](src/model/gpt_mini.py) — 85 lines including generation.

---

## Features

- **Train from the UI** — sliders for steps, context length, `d_model`, layers, heads, batch size; live log stream
- **Ask from the UI** — sliders for top-k passages, max new tokens, temperature, sampling top-k
- **Accelerator auto-detection** — uses MPS (Apple Silicon) → CUDA → CPU in that order
- **Checkpoint save/load** — model config baked into the `.pt` file so it reloads correctly
- **Python-only inference** — no tokenisation libraries, no HuggingFace dependencies; `CharTokenizer` is 30 lines

---

## Tech Stack

| Component | Library |
|---|---|
| Transformer model | PyTorch (built from scratch) |
| TF-IDF retrieval | scikit-learn |
| Web UI | Gradio |
| Chunking / I/O | pure Python |
| Serialisation | joblib (index), torch.save (model) |

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/bakbergen777/smallgpt_ada.git
cd smallgpt_ada
pip install torch gradio scikit-learn joblib

# 2. Add your book
cp path/to/your/book.txt data/raw/book.txt

# 3. Launch the UI
python -m src.ui.app
# → opens http://localhost:7860 in your browser
```

**Ask tab:** Upload `.txt` → click *Load & Build Index* → click *Load checkpoint* → type question → *Ask*

**Train tab:** Set hyperparameters → *Start training* → watch live loss log → checkpoint saved automatically

### CLI inference (no UI)

```bash
python -m src.infer.answer \
  --q "What does the author say about memory?" \
  --k 3 \
  --max_new 200 \
  --temperature 0.9
```

### Train from CLI

```bash
python -m src.train.train \
  --data data/raw/book.txt \
  --steps 800 \
  --ctx 256 \
  --dmodel 256 \
  --layers 4 \
  --heads 4
```

---

## Project Layout

```
smallgpt_ada/
  src/
    model/
      gpt_mini.py         # MiniGPT: SelfAttention, Block, generate()
    train/
      train.py            # training loop + checkpoint save
    infer/
      answer.py           # TF-IDF search + GPT generation (CLI)
    data/
      chunk.py            # text cleaning + overlap chunking
      tfidf_index.py      # build + search TF-IDF index
    ui/
      app.py              # Gradio two-tab app (Ask + Train)
      cli.py              # interactive CLI fallback
    util/
      tokenizer.py        # CharTokenizer (encode / decode)
      io.py               # safe file reader
  data/
    raw/                  # put your book.txt here
    processed/            # chunks.jsonl (auto-generated)
    indices/              # tfidf.joblib (auto-generated)
  checkpoints/            # minigpt.pt (auto-saved after training)
```

---

## What I Learned

- **Building a transformer from scratch** — implementing `SelfAttention` with manual QKV split, causal mask, and scaled dot-product made the GPT-2 paper's equations feel concrete rather than abstract
- **Weight tying** — sharing the input embedding and output projection matrix (`head.weight = tok_emb.weight`) halves parameters with no loss in quality at small scale
- **RAG without a vector DB** — TF-IDF cosine similarity is surprisingly effective for factual book Q&A; the retrieval step does most of the heavy lifting; the GPT just formats the answer
- **Apple Silicon training** — `torch.backends.mps.is_available()` makes local GPU training practical on a MacBook; 800 steps on ~300 KB of text takes ~2 minutes on M-series chip
- **Gradio subprocess streaming** — running the training script in a subprocess and streaming its stdout line-by-line to the UI taught me how to bridge a blocking CLI process into an async web UI

---

## My Role

Sole author — transformer architecture, training loop, TF-IDF retrieval pipeline, Gradio UI, CLI interface, tokeniser, chunker.
