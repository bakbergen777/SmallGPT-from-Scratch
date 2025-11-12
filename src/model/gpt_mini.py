import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, ctx):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        self.register_buffer("mask", mask==1)

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.qkv(x).split(C, dim=-1)
        def split(t): return t.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        q,k,v = map(split, (q,k,v))
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:T,:T], float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, ctx, pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.drop = nn.Dropout(pdrop)
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class MiniGPT(nn.Module):
    # Принимаем и aliases: layers/heads ИЛИ n_layers/n_heads
    def __init__(self, vocab, ctx=256, d_model=256, n_layers=4, n_heads=4, **kwargs):
        super().__init__()
        if "layers" in kwargs: n_layers = kwargs["layers"]
        if "heads"  in kwargs: n_heads  = kwargs["heads"]
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ctx) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None,:,:]
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=1.0, top_k=None):
        for _ in range(max_new):
            idx_cond = idx[:, -self.ctx:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))  # ✅ фикс: k не больше размера словаря
                v,_ = torch.topk(logits, k)
                logits[logits < v[:,[-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
