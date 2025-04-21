import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login
class LlamaAgent:
    def __init__(self):
        # TODO: Choose a new verison of LLama and get it to work.
        model_id = "meta-llama/Llama-2-3b-hf"
        models_dir = "./models"
        model_path = os.path.join(models_dir, "Llama-2-3b-hf")

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            print(f"Downloading {model_id} to {model_path}...")
            hf_token = input("Please enter your HuggingFace token (or set HUGGINGFACE_TOKEN env var): ") or os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ValueError("HuggingFace token is required to download LLaMA models")
            # Login to HuggingFace
            login(token=hf_token)
            snapshot_download(repo_id=model_id, local_dir=model_path)
            print("Download complete.")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def ask(self, query, max_length=None):
        if max_length is None:
            max_length = 2048


        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
