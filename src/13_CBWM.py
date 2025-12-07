import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_classic.memory import ConversationBufferWindowMemory

MODEL_PATH = "./outputs/distilgpt2_finetuned"


class BufferWindowChatbot:
    def __init__(self, model_path: str, k: int = 3):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Window memory: only keeps last k exchanges
        self.memory = ConversationBufferWindowMemory(k=k)

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
        history = self.memory.load_memory_variables({}).get("history", "")

        prompt = f"""You are a small but helpful AI assistant fine-tuned by Ananya.
You remember only the last few messages (window memory).

Recent conversation:
{history}

User: {user_msg}
Assistant:"""

        full_output = self._generate(prompt)
        if "Assistant:" in full_output:
            reply = full_output.split("Assistant:", 1)[1].strip()
        else:
            reply = full_output.strip()

        self.memory.chat_memory.add_user_message(user_msg)
        self.memory.chat_memory.add_ai_message(reply)
        return reply


if __name__ == "__main__":
    bot = BufferWindowChatbot(MODEL_PATH, k=3)
    print("🤖 Buffer Window Memory Chatbot (remembers only last 3 turns). Type 'exit' to quit.\n")

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
