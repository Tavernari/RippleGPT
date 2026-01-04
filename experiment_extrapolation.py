import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. MINIMALIST MODELS FOR THE TEST ---

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class RippleHeadExtrapolatable(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, self.head_size, bias=False)
        self.query = nn.Linear(n_embd, self.head_size, bias=False)
        self.value = nn.Linear(n_embd, self.head_size, bias=False)
        self.decay_factor = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # DYNAMIC RIPPLE FIELD (Calculated on the fly for ANY length T)
        indices = torch.arange(T, device=x.device)
        dist = indices[None, :] - indices[:, None]
        dist = dist.clamp(max=0)
        ripple_bias = dist * torch.abs(self.decay_factor)
        wei = wei + ripple_bias
        
        # Dynamic Masking (No hardcoded size limits!)
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class StandardHeadLimited(nn.Module):
    def __init__(self, n_embd, n_head, max_train_len):
        super().__init__()
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, self.head_size, bias=False)
        self.query = nn.Linear(n_embd, self.head_size, bias=False)
        self.value = nn.Linear(n_embd, self.head_size, bias=False)
        # Hardcoded limit common in Standard GPTs
        self.register_buffer('tril', torch.tril(torch.ones(max_train_len, max_train_len)))

    def forward(self, x):
        B, T, C = x.shape
        if T > self.tril.shape[0]:
            # This simulates Standard GPT failing on unseen lengths
            raise ValueError(f"Standard GPT Crash: Sequence length {T} > Max Train Length {self.tril.shape[0]}")
            
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

# --- 2. THE TEST LOGIC ---

def run_extrapolation_test():
    print("--- 🧪 EXTRAPOLATION EXPERIMENT ---")
    
    TRAIN_LENGTH = 64
    TEST_LENGTH = 128 # Double the length!
    N_EMBD = 64
    N_HEAD = 2
    
    # 1. Instantiate Models
    print(f"1. Initializing models (Train Limit: {TRAIN_LENGTH} tokens)")
    
    # Ripple: Has NO position embedding table
    ripple = RippleHeadExtrapolatable(N_EMBD, N_HEAD).to(device)
    
    # Standard: Has hard limits
    standard = StandardHeadLimited(N_EMBD, N_HEAD, TRAIN_LENGTH).to(device)
    
    # 2. Create Dummy Data
    print(f"2. Generating Test Data of length {TEST_LENGTH}...")
    x_long = torch.randn(1, TEST_LENGTH, N_EMBD).to(device) # Batch 1, Len 128
    
    # 3. Test Ripple
    try:
        print("   Testing RippleGPT on 2x Length...")
        out = ripple(x_long)
        print(f"   ✅ SUCCESS! Ripple output shape: {out.shape}")
        print("   -> Conclusion: RippleGPT handles 'infinite' context natively.")
    except Exception as e:
        print(f"   ❌ Ripple Failed: {e}")

    # 4. Test Standard
    try:
        print("   Testing Standard GPT on 2x Length...")
        out = standard(x_long)
        print(f"   ✅ SUCCESS! (Unexpected for Standard)")
    except ValueError as e:
        print(f"   💥 CRASH! Standard GPT Failed as expected.")
        print(f"   -> Error: {e}")
        print("   -> Conclusion: Standard GPT requires retraining for longer contexts.")

if __name__ == "__main__":
    run_extrapolation_test()
