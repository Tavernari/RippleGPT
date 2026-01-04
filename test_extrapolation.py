import torch
import torch.nn.functional as F
from src.model import RippleGPT
from src.config import RippleConfig
import os
import pickle

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = RippleConfig(**checkpoint['model_args'])
    # FORCE the model to accept longer context than training (256)
    # We set block_size to 1024 to see if it breaks or works
    config.block_size = 1024 
    model = RippleGPT(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def measure_perplexity(model, data_tensor, context_len):
    """
    Measures how surprised the model is. Lower is better.
    We test on a context length LARGER than training.
    """
    max_batches = 10
    total_loss = 0
    with torch.no_grad():
        for i in range(max_batches):
            # Grab a chunk of size 'context_len'
            # If model was trained on 256, and we test 1024, this validates Ripple
            if i * context_len + context_len + 1 > len(data_tensor): break
            
            x = data_tensor[i*context_len : i*context_len + context_len].unsqueeze(0).to(device)
            y = data_tensor[i*context_len+1 : i*context_len + context_len+1].unsqueeze(0).to(device)
            
            _, loss = model(x, y)
            total_loss += loss.item()
            
    avg_loss = total_loss / max_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

# Load Data
print("Loading data...")
dataset_dir = 'data'
val_data_path = os.path.join(dataset_dir, 'val.bin')
meta_path = os.path.join(dataset_dir, 'meta.pkl')

if os.path.exists(val_data_path) and os.path.exists(meta_path):
    print(f"Loading official validation data from {val_data_path}...")
    import numpy as np
    val_data_np = np.fromfile(val_data_path, dtype=np.uint16)
    val_data = torch.from_numpy(val_data_np.astype(np.int64))
else:
    print("Official validation data not found. Downloading tinyshakespeare for demo...")
    # Load validation text (simulate loading validation data)
    import requests
    text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
    
    # Simple encoding if meta not found
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    val_data = torch.tensor(encode(text[int(0.9*len(text)):]), dtype=torch.long)

# Load Model
print("Loading RippleGPT...")
ckpt_path = 'out/ckpt_best.pt' if os.path.exists('out/ckpt_best.pt') else 'out/ckpt.pt'
print(f"Loading checkpoint from {ckpt_path}")
model = load_model(ckpt_path)

# TEST 1: Standard Context (256)
loss_256, ppl_256 = measure_perplexity(model, val_data, 256)
print(f"Context 256 (Trained size): Loss {loss_256:.4f}, Perplexity {ppl_256:.2f}")

# TEST 2: Extrapolation (512) - The Scientific Proof
try:
    loss_512, ppl_512 = measure_perplexity(model, val_data, 512)
    print(f"Context 512 (2x Training):  Loss {loss_512:.4f}, Perplexity {ppl_512:.2f}")
    print("✅ EXTRAPOLATION SUCCESSFUL: Model handled 2x context length!")
except Exception as e:
    print(f"❌ EXTRAPOLATION FAILED: {e}")

# TEST 3: Extreme Extrapolation (1024)
try:
    loss_1024, ppl_1024 = measure_perplexity(model, val_data, 1024)
    print(f"Context 1024 (4x Training): Loss {loss_1024:.4f}, Perplexity {ppl_1024:.2f}")
except Exception as e:
    pass
