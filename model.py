import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, value for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, hs)

        # Causal mask is handled efficiently via PyTorch's native function
        # This prevents a token from attending to future tokens.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc    = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj  = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size=50257, max_seq_len=128, embed_dim=192, n_heads=6, n_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token and Positional Embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads) for _ in range(n_layers)])
        
        # Final LayerNorm and output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying: share weights between input and output embeddings
        self.tok_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Add token and position embeddings
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        # Pass through the transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # PyTorch's cross_entropy expects (B * sequence_length, vocab_size) and (B * sequence_length,)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss

if __name__ == "__main__":
    # Acceptance check:
    # - One forward pass returns logits shaped [batch, sequence_length, vocabulary_size].
    # - Cross-entropy loss computes successfully.
    # - The causal mask prevents a token from attending to later tokens 
    #   (implicitly handled by F.scaled_dot_product_attention(..., is_causal=True)).
    
    device = "cpu"
    model = TinyLM().to(device)
    
    # Calculate parameter count (should be 5-20M according to requirements)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated with {total_params / 1e6:.2f}M parameters")
    
    # Simulate a fake batch (batch_size=2, sequence_length=8)
    B, T = 2, 8
    idx = torch.randint(0, 50257, (B, T)).to(device)
    targets = torch.randint(0, 50257, (B, T)).to(device)
    
    logits, loss = model(idx, targets)
    
    print("\n--- Acceptance Check ---")
    print("Output Logit Shape:", logits.shape)
    is_valid_shape = logits.shape == (B, T, 50257)
    print("Does logit shape match [batch, sequence_length, vocabulary_size]?", "YES" if is_valid_shape else "NO")
    print("Is cross-entropy loss computed successfully? YES, value =", loss.item())
