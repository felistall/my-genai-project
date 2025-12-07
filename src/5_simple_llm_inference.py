from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Use the fine-tuned distilgpt2 model
MODEL_PATH = "./outputs/distilgpt2_finetuned"


def generate_answer(instruction: str, max_new_tokens: int = 80) -> str:
    """
    Build the same style of prompt we used during training:
    'Instruction: ...\\nResponse:'
    Then let the model complete it.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,  # a bit more controlled randomness
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the part after "Response:"
    if "Response:" in full_text:
        answer = full_text.split("Response:", 1)[1].strip()
    else:
        answer = full_text.strip()

    return answer


if __name__ == "__main__":
    print("Tiny fine-tuned distilgpt2 demo. Type 'exit' to quit.\n")

    while True:
        user_instruction = input("Instruction: ")
        if user_instruction.lower() in ["exit", "quit"]:
            break

        reply = generate_answer(user_instruction)
        print("Model:", reply)
        print("-" * 60)
