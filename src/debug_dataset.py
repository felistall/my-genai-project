import os
from pathlib import Path

print("Current working directory:", os.getcwd())

path = Path("data/pretrain_corpus.txt")
print("Exists?", path.exists())
print("Absolute path:", path.resolve())

if path.exists():
    text = path.read_text(encoding="utf-8")
    print("Number of characters in file:", len(text))
    print("First 200 characters preview:")
    print(repr(text[:200]))
