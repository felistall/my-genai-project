from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ✅ Use a real pretrained model instead of tiny scratch model
BASE_MODEL = "distilgpt2"   # you can switch to "gpt2" if your system can handle it

DATA_FILE = "data/finetune_data.jsonl"
OUTPUT_DIR = "./outputs/distilgpt2_finetuned"


def format_example(example):
    """
    Turn each instruction/output pair into a single training string.
    Format:
    Instruction: ...
    Response: ...
    """
    prompt = f"Instruction: {example['instruction']}\nResponse:"
    text = prompt + " " + example["output"]
    return {"text": text}


if __name__ == "__main__":
    # 1) Load dataset from jsonl
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    dataset = dataset.map(format_example)

    print("Number of finetune samples:", len(dataset))

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # 3) Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal language modelling
    )

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,          # you can increase to 10 if you add more data
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # 6) Train
    trainer.train()

    # 7) Save final finetuned model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Fine-tuning finished successfully. Saved to:", OUTPUT_DIR)
