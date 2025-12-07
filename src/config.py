# config.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    raise ValueError("Set HUGGINGFACE_TOKEN in .env")

# Call this once in any script that needs login
def hf_login():
    login(token=HF_TOKEN)
