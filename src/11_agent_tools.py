# src/11_agent_tools.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

MODEL_PATH = "./outputs/distilgpt2_finetuned"


def calculator_tool(expression: str) -> str:
    """
    Very simple calculator tool.
    Example: '2 + 3 * 4'
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def call_llm(prompt: str, tokenizer, model, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,   # 🔹 reduce repetition
            no_repeat_ngram_size=3,   # 🔹 no repeating 3-word phrases
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def looks_like_math(expr: str) -> bool:
    """
    Heuristic: returns True if the user input looks like a simple math expression.
    e.g. "3+2-8", "10 * 5", "(2+3)/4"
    """
    expr = expr.replace(" ", "")
    return bool(re.fullmatch(r"[0-9+\-*/().]+", expr))


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Simple agent demo. Type 'exit' to quit.")
    print("Type things like 'calculate 2 + 3 * 4' or just '2+3*4' or normal questions.\n")

    while True:
        user = input("User: ")
        if user.strip() == "":
            continue
        if user.lower() in ["exit", "quit"]:
            break

        lower = user.lower().strip()

        # 1) If starts with 'calculate ', use tool
        if lower.startswith("calculate "):
            expr = user[len("calculate "):].strip()
            tool_result = calculator_tool(expr)
            prompt = (
                "You are an assistant that explains calculator results clearly.\n\n"
                f"User expression: {expr}\n"
                f"Tool result: {tool_result}\n\n"
                "Explain this result to the user in simple language:"
            )
            reply = call_llm(prompt, tokenizer, model)

        # 2) If it looks like a math expression (e.g. '3+2-8'), use tool too
        elif looks_like_math(user):
            expr = user.strip()
            tool_result = calculator_tool(expr)
            prompt = (
                "You are an assistant that explains calculator results clearly.\n\n"
                f"User expression: {expr}\n"
                f"Tool result: {tool_result}\n\n"
                "Explain this result to the user in simple language:"
            )
            reply = call_llm(prompt, tokenizer, model)

        # 3) Otherwise, normal LLM chat
        else:
            prompt = f"User: {user}\nAssistant:"
            reply = call_llm(prompt, tokenizer, model)

        print("Agent:", reply)
        print("-" * 60)
