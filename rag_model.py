import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGModel:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        logging.info(f"Loading the tokenizer and model for '{model_name}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)  # Define self.model
        if torch.backends.mps.is_available():
            logging.info("MPS is available. Using Apple Silicon GPU for inference.")
            self.device = torch.device("mps")
        else:
            logging.info("MPS not available. Using CPU for inference.")
            self.device = torch.device("cpu")
        self.model.to(self.device)  # Move the model to the device
        logging.info(f"Model moved to device: {self.device}")
        # Set tokenizer pad token to EOS token (if your tokenizer does not have a default pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Model and tokenizer are successfully loaded.")

    def generate_response(self, input_text, context_texts, max_new_tokens=10):
        separator = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else ' '
        combined_input = f"{input_text} {separator.join(context_texts)}"

        # Encode the combined input and generate attention mask
        encoding = self.tokenizer.encode_plus(
            combined_input,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Adjust based on your model's maximum input size
            padding="max_length",  # Pad to max_length
            return_attention_mask=True
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():  
        # Generate the response using various techniques to improve quality and reduce repetition
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512 + max_new_tokens,  # Ensure max_length accounts for the input and new tokens
                do_sample=True,  # Enable sampling for more varied outputs
                num_beams=5,  # Enable Beam Search
                early_stopping=True,  # Enable Early Stopping
                no_repeat_ngram_size=10,  # Enable No Repeat N-Gram Penalty to avoid repeating 2-grams
                top_k=30,  # Enable Top-K Sampling
                top_p=0.7,  # Enable Top-p (Nucleus) Sampling with p=0.92
                temperature=0.7,  # Adjust Temperature to control randomness
                repetition_penalty=1.2  # Use repetition_penalty
            )

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response
