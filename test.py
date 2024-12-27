import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

# Load the base model and tokenizer
base_model_path = "/model-weights/Llama-3.2-1B"
lora_model_path = "./llama2-qa-lora-final"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

def generate_response(text, question, max_length=512):
    # Format the prompt
    prompt = f"### Text:\n{text}\n### Question:\n{question}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    return response

# Load test cases from CSV file
test_cases = pd.read_csv('/h/emzed/data/qa_discharge_masked.csv', nrows=1)

# Run tests
test_cases = pd.read_csv('/h/emzed/data/qa_discharge_masked.csv', nrows=1)
