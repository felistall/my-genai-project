# src/15_summary_window_memory.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./outputs/distilgpt2_finetuned"


class SummaryWindowMemory:
    """
    Summary of full conversation + last N raw turns.
    Similar idea to ConversationSummaryBufferMemory.
    """

    def __init__(self, window_size: int = 3):
        self.turns = []
        self.summary = ""
        self.window_size = window_size

    def add(self, user_msg: str, bot_reply: str):
        self.turns.append((user_msg, bot_reply))

    def full_history_text(self) -> str:
        lines = []
        for u, b in self.turns:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {b}")
        return "\n".join(lines)

    def recent_history_text(self) -> str:
        lines = []
        for u, b in self.turns[-self.window_size:]:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {b}")
        return "\n".join(lines)

    def should_summarize(self) -> bool:
        return len(self.turns) >= 4


class SummaryWindowChatbot:
    def __init__(self, model_path: str, window_size: int = 3):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.memory = SummaryWindowMemory(window_size=window_size)

    def _generate_raw(self, prompt: str, max_new_tokens: int = 120) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _summarize_if_needed(self):
        if self.memory.should_summarize():
            history = self.memory.full_history_text()
            prompt = f"""Summarize the following conversation in a few sentences:

{history}

Summary:"""
            full = self._generate_raw(prompt, max_new_tokens=80)
            if "Summary:" in full:
                self.memory.summary = full.split("Summary:", 1)[1].strip()
            else:
                self.memory.summary = full.strip()

    def chat(self, user_msg: str) -> str:
        self._summarize_if_needed()

        summary = self.memory.summary
        recent = self.memory.recent_history_text()

        prompt = f"""You are a small but helpful AI assistant fine-tuned by Ananya.
Use both the long-term summary and the recent turns to answer.

Long-term summary:
{summary if summary else "(no summary yet)"}

Recent conversation:
{recent}

User: {user_msg}
Assistant:"""

        full_output = self._generate_raw(prompt)
        if "Assistant:" in full_output:
            reply = full_output.split("Assistant:", 1)[1].strip()
        else:
            reply = full_output.strip()

        self.memory.add(user_msg, reply)
        return reply


if __name__ == "__main__":
    bot = SummaryWindowChatbot(MODEL_PATH, window_size=3)
    print("🤖 Summary+Window Memory Chatbot. Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ["exit", "quit"]:
            print("Bot: Bye! 👋")
            break

        reply = bot.chat(user)
        print("Bot:", reply)
        print("-" * 80)
