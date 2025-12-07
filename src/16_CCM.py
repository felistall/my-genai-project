# src/16_custom_memory_chat.py

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./outputs/distilgpt2_finetuned"


class CustomProfileMemory:
    """
    Custom memory that extracts:
    - user name
    - interests
    - goals
    - mood / stress words
    plus raw chat history.
    """

    def __init__(self):
        self.name = None
        self.interests = set()
        self.goals = set()
        self.mood_tags = set()
        self.turns = []

    def add_turn(self, user_msg: str, bot_reply: str):
        self.turns.append((user_msg, bot_reply))
        self._update_profile(user_msg)

    def _update_profile(self, text: str):
        lower = text.lower()

        # Name detection
        m = re.search(r"my name is ([a-zA-Z ]+)", lower)
        if m:
            self.name = m.group(1).strip().title()

        # Interests
        keywords_interests = ["ai", "ml", "gen ai", "coding", "math", "music", "games"]
        for k in keywords_interests:
            if k in lower:
                self.interests.add(k)

        # Goals
        if "placement" in lower or "placements" in lower:
            self.goals.add("crack placements")
        if "exam" in lower or "exams" in lower:
            self.goals.add("do well in exams")
        if "project" in lower:
            self.goals.add("finish good projects")

        # Mood / stress
        stress_words = ["stressed", "anxious", "tired", "overwhelmed", "confused"]
        for w in stress_words:
            if w in lower:
                self.mood_tags.add(w)

    def profile_text(self) -> str:
        lines = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.interests:
            lines.append(f"Interests: {', '.join(sorted(self.interests))}")
        if self.goals:
            lines.append(f"Goals: {', '.join(sorted(self.goals))}")
        if self.mood_tags:
            lines.append(f"Current mood tags: {', '.join(sorted(self.mood_tags))}")
        return "\n".join(lines) if lines else "(no profile yet)"

    def history_text(self, last_n: int = 5) -> str:
        lines = []
        for u, b in self.turns[-last_n:]:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {b}")
        return "\n".join(lines)


class CustomMemoryChatbot:
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.memory = CustomProfileMemory()

    def _generate(self, prompt: str, max_new_tokens: int = 120) -> str:
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

    def chat(self, user_msg: str) -> str:
        profile = self.memory.profile_text()
        recent = self.memory.history_text(last_n=5)

        prompt = f"""You are a small but caring AI assistant fine-tuned by Ananya.
Use the user profile and recent chat to answer in a personal, supportive way.

User profile:
{profile}

Recent conversation:
{recent}

User: {user_msg}
Assistant:"""

        full_output = self._generate(prompt)
        if "Assistant:" in full_output:
            reply = full_output.split("Assistant:", 1)[1].strip()
        else:
            reply = full_output.strip()

        self.memory.add_turn(user_msg, reply)
        return reply


if __name__ == "__main__":
    bot = CustomMemoryChatbot(MODEL_PATH)
    print("🤖 Custom Profile Memory Chatbot. Type 'exit' to quit.\n")

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
