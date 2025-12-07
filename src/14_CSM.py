# src/14_summary_memory.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./outputs/distilgpt2_finetuned"


class SimpleSummaryMemory:
    """
    Keeps full history + a running summary that is updated using the LLM.
    This is conceptually similar to LangChain's ConversationSummaryMemory.
    """

    def __init__(self):
        self.turns = []   # list of (user, bot)
        self.summary = ""  # running textual summary

    def add(self, user_msg: str, bot_reply: str):
        self.turns.append((user_msg, bot_reply))

    def get_history_text(self) -> str:
        lines = []
        for u, b in self.turns:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {b}")
        return "\n".join(lines)

    def should_summarize(self) -> bool:
        # summarize after every 4 turns, for example
        return len(self.turns) >= 4


class SummaryChatbot:
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.memory = SimpleSummaryMemory()

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

    def _summarize_history(self):
        history_text = self.memory.get_history_text()
        prompt = f"""Summarize the following conversation briefly, focusing on key facts and topics:

{history_text}

Summary:"""

        full = self._generate_raw(prompt, max_new_tokens=80)
        if "Summary:" in full:
            summary = full.split("Summary:", 1)[1].strip()
        else:
            summary = full.strip()
        self.memory.summary = summary

    def chat(self, user_msg: str) -> str:
        # Optionally update summary
        if self.memory.should_summarize():
            self._summarize_history()

        summary = self.memory.summary
        history_text = self.memory.get_history_text()

        prompt = f"""You are a small but helpful AI assistant fine-tuned by Ananya.
Use the summary and conversation below to respond briefly and clearly.

Current summary:
{summary if summary else "(no summary yet)"}

Full conversation:
{history_text}

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
    bot = SummaryChatbot(MODEL_PATH)
    print("🤖 Summary Memory Chatbot (keeps + updates a summary). Type 'exit' to quit.\n")

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
