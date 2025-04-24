import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
class LlamaAgent:
    def __init__(self):
        model_id = "meta-llama/Llama-3.1-8B"
        models_dir = "./models"
        model_path = os.path.join(models_dir, "Meta-Llama-3.1-8B")

        os.makedirs(models_dir, exist_ok=True)
        if not os.path.exists(model_path):
            print(f"Downloading {model_id} to {model_path}...")
            hf_token = input("Please enter your HuggingFace token (or set HUGGINGFACE_TOKEN env var): ") or os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ValueError("HuggingFace token is required to download LLaMA models")

            login(token=hf_token)
            snapshot_download(repo_id=model_id, local_dir=model_path)
            print("Download complete.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check for GPU availabilityma
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            print("GPU not available, using CPU")
            self.device = torch.device("cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def ask(self, query, max_length=None):
        if max_length is None:
            max_length = 128

        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.2,
            top_p=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def trim_context(self, context):
        if not context:
            return []

        max_tokens = 100000
        trimmed_context = []
        current_tokens = 0

        for item in reversed(context):
            tokens = len(self.tokenizer.encode(item))
            if current_tokens + tokens > max_tokens:
                break
            trimmed_context.append(item)
            current_tokens += tokens

        return list(reversed(trimmed_context))
