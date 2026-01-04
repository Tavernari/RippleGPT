import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import RippleConfig

class RippleHead(nn.Module):
    def __init__(self, config: RippleConfig):
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Learnable Decay (The "Magnet")
        self.decay_factor = nn.Parameter(torch.tensor([-0.8]))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Base Affinity
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Ripple Field (Computed dynamically for ANY length T)
        indices = torch.arange(T, device=x.device)
        dist = indices[None, :] - indices[:, None]
        dist = dist.clamp(max=0) # Causal
        
        ripple_bias = dist * torch.abs(self.decay_factor)
        wei = wei + ripple_bias
        
        # Causal Mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(mask == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        return wei @ v

class RippleMLP(nn.Module):
    def __init__(self, config: RippleConfig):
        super().__init__()
        # Parameter Efficiency Logic: 8/3 ratio to match Standard GPT params
        hidden_dim = int(config.n_embd * 8 / 3)
        if hidden_dim % 2 != 0:
            hidden_dim += 1
            
        self.fc1 = nn.Linear(config.n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim // 2, config.n_embd) # Returns from split
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.fc1(x)
        x_val, x_gate = h.chunk(2, dim=-1)
        # Gated Multiplicative Interaction
        return self.dropout(self.fc2(x_val * F.silu(x_gate)))

class Block(nn.Module):
    def __init__(self, config: RippleConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.heads = nn.ModuleList([RippleHead(config) for _ in range(config.n_head)])
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = RippleMLP(config)

    def forward(self, x):
        # Parallel Heads
        heads_out = torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1)
        x = x + heads_out
        x = x + self.ffwd(self.ln2(x))
        return x

class RippleGPT(nn.Module):
    def __init__(self, config: RippleConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        
        if config.use_absolute_pos_emb:
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        x = self.token_embedding_table(idx)
        
        if self.config.use_absolute_pos_emb:
            pos = torch.arange(T, device=device)
            x = x + self.position_embedding_table(pos)
            
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B*T, C)
            flat_targets = targets.view(B*T)
            loss = F.cross_entropy(flat_logits, flat_targets)
        return logits, loss
    
    # HuggingFace compatibility: Number of parameters
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size ONLY IF we are using pos embs
            if self.config.use_absolute_pos_emb:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                # If we are relying on Ripple Field, we can technically feed everything
                # BUT for efficiency we usually crop significantly past training context?
                # Actually, the prompt says "it should be able to handle longer texts". 
                # Let's keep all context to prove extrapolation unless it OOMs.
                idx_cond = idx

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
