import os
import torch
from huggingface_hub import HfApi, create_repo
from src.model import RippleGPT, RippleConfig
from dotenv import load_dotenv

# 1. Configuration
load_dotenv() # Load variables from .env

hf_username = os.getenv("HF_USERNAME")
token = os.getenv("HF_TOKEN")

if not hf_username or not token:
    raise ValueError("Error: HF_USERNAME or HF_TOKEN not found in environment variables (or .env file).")

repo_id = f"{hf_username}/RippleGPT-Nano"

# 2. Create Repo
print(f"Creating repo: {repo_id}...")
api = HfApi(token=token)
try:
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
except Exception as e:
    print(f"Note: Repo creation might have failed or already exists: {e}")

# 3. Upload Code and Model
print("Uploading files...")
try:
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".git/*", ".venv/*", "__pycache__/*", "*.DS_Store", "out/*", ".env"]
    )
    print(f"✅ SUCCESS! Model is live at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
