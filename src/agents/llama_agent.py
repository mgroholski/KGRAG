import os
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
class LlamaAgent:
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3-8B"
        models_dir = "./models"
        model_path = os.path.join(models_dir, "Meta-Llama-3-8B")

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

        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Using GPU: {gpus[0]}")
            # Set memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("GPU not available, using CPU")

        self.model = TFAutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=tf.float16
        )

    def ask(self, query, max_length=None):
        if max_length is None:
            max_length = 512

        inputs = self.tokenizer(query, return_tensors="tf")
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
