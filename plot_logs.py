import matplotlib.pyplot as plt
import re
import os

train_loss = []
val_loss = []
steps = []
val_steps = []

log_file_path = 'training.log'
if not os.path.exists(log_file_path):
    print(f"File {log_file_path} not found. Please paste your training logs into this file first.")
    # Create dummy if not exists for demo purposes so it doesn't crash immediately
    # In real usage the user should provide the log
    exit(1)

with open(log_file_path, 'r') as f:
    for line in f:
        # Parse training loss
        # "iter 100: loss 1.8030"
        if "iter" in line and "loss" in line and "time" in line:
            parts = line.split()
            # Handling formatting variability
            try:
                # expected format: iter 100: loss 1.8030, time ...
                step_idx = parts.index('iter') + 1
                loss_idx = parts.index('loss') + 1
                
                step = int(parts[step_idx].replace(':', ''))
                loss = float(parts[loss_idx].replace(',', ''))
                train_loss.append(loss)
                steps.append(step)
            except ValueError:
                continue
        
        # Parse validation loss
        # "step 500: train loss 1.2915, val loss 1.5601"
        if "step" in line and "val loss" in line:
            parts = line.split()
            try:
                step_idx = parts.index('step') + 1
                val_loss_idx = parts.index('val') + 2 # "val loss X"
                
                step = int(parts[step_idx].replace(':', ''))
                v_loss = float(parts[val_loss_idx])
                val_loss.append(v_loss)
                val_steps.append(step)
            except ValueError:
                continue

if not steps:
    print("No data parsed. Check log format.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(steps, train_loss, label='Train Loss', alpha=0.6)
if val_steps:
    plt.plot(val_steps, val_loss, label='Validation Loss', linewidth=3, color='red')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('RippleGPT Training Dynamics: Identifying Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png')
print("Plot saved to loss_curve.png")
# plt.show() # Disabled for headless environment
