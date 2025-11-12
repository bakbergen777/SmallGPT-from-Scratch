class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {c:i for i,c in enumerate(chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    def encode(self, s: str): return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids): return ''.join(self.itos[i] for i in ids if i in self.itos)
