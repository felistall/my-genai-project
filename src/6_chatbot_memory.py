# src/6_chatbot_memory.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./outputs/distilgpt2_finetuned"


class SimpleChatbot:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.history = []  # list of (user, bot) tuples

    def build_prompt(self, user_input, max_turns=5):
        history_text = ""
        for u, b in self.history[-max_turns:]:
            history_text += f"User: {u}\nBot: {b}\n"
        history_text += f"User: {user_input}\nBot:"
        return history_text

    def chat(self, user_input, max_new_tokens=80):
        prompt = self.build_prompt(user_input)
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

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Bot:" in full_text:
            bot_reply = full_text.split("Bot:")[-1].strip()
        else:
            bot_reply = full_text.strip()

        self.history.append((user_input, bot_reply))
        return bot_reply


if __name__ == "__main__":
    bot = SimpleChatbot(MODEL_PATH)
    print("Chatbot with simple memory. Type 'exit' to quit.\n")

    while True:
        user = input("You: ")
        if user.lower() in ["exit", "quit"]:
            break
        reply = bot.chat(user, max_new_tokens=40)
        print("Bot:", reply)
        print("-" * 60)
