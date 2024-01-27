import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TinyLlama:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            load_in_4bit=True,
            device_map="auto",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"LLM loaded to {self.model.device}")

        self._messages = []

    def __call__(self, messages, *args, **kwds):
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(tokenized_chat, return_tensors="pt").to(
            self.model.device
        )

        outputs = self.model.generate(
            **inputs,
            use_cache=True,
            max_length=1000,
            min_length=10,
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
