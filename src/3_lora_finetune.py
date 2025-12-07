from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

BASE_MODEL = "distilgpt2"                      # base pretrained model
DATA_FILE = "data/finetune_data.jsonl"
OUTPUT_DIR = "./outputs/distilgpt2_lora"


def format_example(example):
    prompt = f"Instruction: {example['instruction']}\nResponse:"
    text = prompt + " " + example["output"]
    return {"text": text}


def tokenize_function(example, tokenizer, max_length=128):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


if __name__ == "__main__":
    # 1) Load dataset
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    dataset = dataset.map(format_example)
    print("Number of LoRA finetune samples:", len(dataset))

    # 2) Tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # 3) Wrap with LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ✅ 4) Data collator: this will create `labels` from `input_ids`
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,   # ✅ important
    )

    # 7) Train
    trainer.train()

    # 8) Save LoRA adapter weights
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ LoRA fine-tuning finished. Saved adapter to:", OUTPUT_DIR)
