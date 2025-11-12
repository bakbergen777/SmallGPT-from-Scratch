import argparse, torch, torch.nn as nn
from pathlib import Path
from src.util.io import read_text
from src.util.tokenizer import CharTokenizer
from src.model.gpt_mini import MiniGPT

def device_str():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def make_batch(data_ids, ctx, bs):
    ix = torch.randint(0, len(data_ids)-ctx-1, (bs,))
    x = torch.stack([torch.tensor(data_ids[i:i+ctx]) for i in ix])
    y = torch.stack([torch.tensor(data_ids[i+1:i+ctx+1]) for i in ix])
    return x, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/book.txt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="checkpoints/minigpt.pt")
    args = ap.parse_args()

    text = read_text(args.data)
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(ids) < args.ctx + 2:
        raise SystemExit("Training text too short for ctx. Add more text to data/raw/book.txt")

    split = int(0.9*len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    dev = device_str()
    model = MiniGPT(tok.vocab_size, args.ctx, args.dmodel, args.layers, args.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    Path("checkpoints").mkdir(exist_ok=True)

    model.train()
    for step in range(args.steps):
        x,y = make_batch(train_ids, args.ctx, args.bs)
        x,y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                vx, vy = make_batch(val_ids, args.ctx, args.bs)
                vx, vy = vx.to(dev), vy.to(dev)
                vloss = loss_fn(model(vx).view(-1, tok.vocab_size), vy.view(-1))
            print(f"step {step:4d} | loss {loss.item():.3f} | val {vloss.item():.3f}")

    torch.save({
        "model": model.state_dict(),
        "vocab_size": tok.vocab_size,
        "stoi": tok.stoi,
        "itos": tok.itos,
        "cfg": dict(ctx=args.ctx, d_model=args.dmodel, layers=args.layers, heads=args.heads)
    }, args.ckpt)
    print(f"✅ Saved checkpoint → {args.ckpt}")

    start = torch.tensor([[tok.stoi.get(' ', 0)]], device=dev)
    out = model.generate(start, max_new=200, temperature=0.9, top_k=50)
    print("=== SAMPLE ===")
    print(tok.decode(out[0].tolist()))
