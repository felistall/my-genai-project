from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import os


MODEL_NAME = "tiny-gpt2-pretrained"


def load_dataset_from_txt(file_path, tokenizer, block_size=32):
    """
    Read plain text from file, tokenize it, and split into chunks.
    Works even if text is short. Uses all tokens (no silent drop).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(
            "The corpus file is empty. Please add some text in data/pretrain_corpus.txt"
        )

    # Tokenize full text once
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = tokenized["input_ids"][0]
    print("Total number of tokens in corpus:", len(input_ids))

    # Chunk into blocks of block_size tokens
    chunks = []
    for i in range(0, len(input_ids), block_size):
        chunk = input_ids[i : i + block_size]
        # keep even short last chunk
        if len(chunk) == block_size:
            chunks.append(chunk)

    if len(chunks) == 0:
        raise ValueError(
            "After chunking, no samples were created. "
            "Try adding more text or reducing block_size."
        )

    print("Number of chunks (samples):", len(chunks))

    # Build a Hugging Face Dataset
    return Dataset.from_dict({"input_ids": [c.tolist() for c in chunks]})


def collate_fn(batch, pad_token_id):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),  # LM target = input
    }


if __name__ == "__main__":
    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Tiny GPT-2 config (small for CPU)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)

    # 3) Load dataset from your text file
    dataset = load_dataset_from_txt("D:\my-genai-project\data\pretrain_corpus.txt", tokenizer, block_size=32)

    # 4) Training settings
    training_args = TrainingArguments(
        output_dir="./outputs/tiny_pretrained",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    # 6) Train!
    trainer.train()
    trainer.save_model("./outputs/tiny_pretrained")
    tokenizer.save_pretrained("./outputs/tiny_pretrained")

    print("✅ Pre-training finished successfully.")
