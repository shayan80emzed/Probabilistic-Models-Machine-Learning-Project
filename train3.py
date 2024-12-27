import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import pandas as pd

print(torch.cuda.is_available())

# Load the model and tokenizer
model_path = "/model-weights/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load and prepare the dataset
df = pd.read_csv("/h/emzed/data/qa_discharge_masked.csv")
dataset = Dataset.from_pandas(df)

# Add dataset splitting
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Add this near the top of the file, with other configuration variables
MAX_SEQ_LENGTH = 1024  # Adjust this value as needed

def tokenize_function(examples):
    # Prepare the input prompt and expected output
    prompts = [f"### Text:\n{text}\n### Question:\n" for text in examples['masked_text']]
    targets = [f"{question}" + tokenizer.eos_token for question in examples['q']]  # The target question
    
    # Tokenize the input prompt and target output
    inputs = tokenizer(prompts, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
    outputs = tokenizer(targets, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
    
    # Combine inputs and outputs into a single dataset
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Update tokenization to handle both splits
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
eval_tokenized = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./emziiiii-llama2-qa-lora",
    report_to="none",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps=None,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train the model
print("Training model...")
trainer.train()

# Save the trained model
model.save_pretrained("./llama2-qa-lora-final")
