import os
import time
import math
import pickle
import numpy as np
import torch
from src.model import RippleGPT
from src.config import RippleConfig

# -----------------------------------------------------------------------------
# Default Configuration
out_dir = 'out'
eval_interval = 250 # keep frequent for demo
log_interval = 10
eval_iters = 200
always_save_checkpoint = False # if True, always save a checkpoint after each eval

dataset = 'shakespeare_char'
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary for small networks, but good practice

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'        
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# -----------------------------------------------------------------------------

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join('data', 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join('data', 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size].astype(np.int64))) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size].astype(np.int64))) for i in ix])
    
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# estimation of loss
@torch.no_grad()
def estimate_loss(model, ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)

    # attempt to load meta from data directory
    meta_path = os.path.join('data', 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      dropout=dropout, vocab_size=meta_vocab_size if meta_vocab_size is not None else 65)
    
    gptconf = RippleConfig(**model_args)
    model = RippleGPT(gptconf)
    
    # print parameter count
    print(f"Number of parameters: {model.get_num_params()/1e6:.2f}M")
    
    model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, beta2))

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches
    from contextlib import nullcontext
    ctx = nullcontext() if device == 'cpu' or device == 'mps' else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    
    iter_num = 0
    best_val_loss = 1e9

    while iter_num < max_iters:

        # determine and set the learning rate for this iteration
        lr = learning_rate # simple constant lr for now or implement schedule
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            is_best = losses['val'] < best_val_loss
            if is_best:
                best_val_loss = losses['val']
            
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': gptconf,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
                if is_best:
                    print(f"saving best checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt_best.pt'))

                # Also save state dict for HF
                torch.save(model.state_dict(), "ripplegpt_state.pt")

        # forward backward update
        with ctx:
            logits, loss = model(X, Y)
        
        # backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
        iter_num += 1
        X, Y = get_batch('train')

if __name__ == '__main__':
    main()
