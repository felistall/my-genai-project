# src/16_full_chatbot_with_memory.py

import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_classic.memory import ConversationBufferMemory

MODEL_PATH = "./outputs/distilgpt2_finetuned"
PROFILE_PATH = "./outputs/profile_memory.json"


class CustomProfileMemory:
    """
    Stores structured info about the user (name, etc.)
    + keeps raw chat history via LangChain's ConversationBufferMemory.
    """

    def __init__(self):
        self.name = None
        self.buffer_memory = ConversationBufferMemory()  # chat history

    # ---------- profile logic ----------

    def update_from_user_text(self, text: str):
        lower = text.lower()

        # simple name extraction: "my name is Ananya"
        m = re.search(r"my name is ([a-zA-Z ]+)", lower)
        if m:
            self.name = m.group(1).strip().title()

    def to_dict(self):
        return {
            "name": self.name,
            "history": self.buffer_memory.load_memory_variables({}).get("history", ""),
        }

    def load_from_dict(self, data: dict):
        self.name = data.get("name")
        history_text = data.get("history", "")
        # rebuild chat history into buffer_memory if you want (optional)
        # for simplicity we just store future messages; history here is mostly for debugging

    # ---------- history helpers ----------

    def add_turn(self, user_msg: str, bot_reply: str):
        self.buffer_memory.chat_memory.add_user_message(user_msg)
        self.buffer_memory.chat_memory.add_ai_message(bot_reply)
        self.update_from_user_text(user_msg)

    def history_text(self) -> str:
        return self.buffer_memory.load_memory_variables({}).get("history", "")


class TinyChatbotWithProfileMemory:
    def __init__(self, model_path: str, profile_path: str):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.profile_path = profile_path
        self.memory = CustomProfileMemory()
        self._load_profile_from_disk()

    def _load_profile_from_disk(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory.load_from_dict(data)
                print(f"✅ Loaded profile from {self.profile_path}: name={self.memory.name}")
            except Exception as e:
                print(f"⚠️ Could not load profile: {e}")

    def save_profile_to_disk(self):
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        data = self.memory.to_dict()
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Profile saved to {self.profile_path}")

    def _generate_llm(self, prompt: str, max_new_tokens: int = 120) -> str:
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
        lower = user_msg.lower()

        # 🔹 1) If user asks about their name and we have it, answer directly
        if ("what is my name" in lower or "do you remember my name" in lower) and self.memory.name:
            reply = f"You told me earlier that your name is {self.memory.name} 😊"
            # still log in memory
            self.memory.add_turn(user_msg, reply)
            return reply

        # 🔹 2) Normal LLM-based reply using history + profile
        history = self.memory.history_text()
        profile_info = f"User name: {self.memory.name}" if self.memory.name else "(no name stored yet)"

        prompt = f"""You are a small but helpful AI assistant fine-tuned by Ananya.
You explain things in simple, clear language, like talking to a college friend.

User profile:
{profile_info}

Conversation so far:
{history}

User: {user_msg}
Assistant:"""

        full_output = self._generate_llm(prompt)

        if "Assistant:" in full_output:
            reply = full_output.split("Assistant:", 1)[1].strip()
        else:
            reply = full_output.strip()

        # 🔹 3) Update both chat history and profile from this turn
        self.memory.add_turn(user_msg, reply)

        return reply


def main():
    bot = TinyChatbotWithProfileMemory(MODEL_PATH, PROFILE_PATH)

    print("\n🤖 Tiny Chatbot with Profile + Conversation Memory")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ["exit", "quit"]:
                print("Bot: Bye! 👋 Saving your profile...")
                bot.save_profile_to_disk()
                break

            reply = bot.chat(user)
            print("Bot:", reply)
            print("-" * 80)
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving profile and exiting...")
        bot.save_profile_to_disk()


if __name__ == "__main__":
    main()
