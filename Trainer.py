from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
import os
import random
import logging
from glob import glob
from os.path import join as pj

class Llama2Trainer:
    def __init__(self,
                 checkpoint_dir: str,
                 model_dir: str = "/model-weights/Llama-2-7b-hf",
                 dataset_path: str = "lmqg/qg_squad",
                 dataset_name: str = 'default',
                 max_length: int = 512,
                 max_length_output: int = 128,
                 epoch: int = 3,
                 batch: int = 4,
                 lr: float = 3e-5,
                 fp16: bool = True,
                 random_seed: int = 42,
                 gradient_accumulation_steps: int = 4,
                 logging_dir: str = './logs'):
        
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.epoch = epoch
        self.batch = batch
        self.lr = lr
        self.fp16 = fp16
        self.random_seed = random_seed
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Set seeds for reproducibility
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        # Logger setup
        os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            filename=pj(logging_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s'
        )
        logging.info('Initialized Llama2Trainer')

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_data(self, dataset):
        def tokenize_function(example):
            inputs = self.tokenizer(example['text'], max_length=self.max_length, truncation=True, padding="max_length")
            outputs = self.tokenizer(example['question'], max_length=self.max_length_output, truncation=True, padding="max_length")
            inputs["labels"] = outputs["input_ids"]
            return inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_dataset, eval_dataset=None):
        # Tokenize datasets
        logging.info("Tokenizing datasets")
        train_dataset = self.preprocess_data(train_dataset)
        if eval_dataset:
            eval_dataset = self.preprocess_data(eval_dataset)

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            evaluation_strategy="steps" if eval_dataset else "no",
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            per_device_train_batch_size=self.batch,
            per_device_eval_batch_size=self.batch if eval_dataset else None,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epoch,
            learning_rate=self.lr,
            fp16=self.fp16,
            seed=self.random_seed,
            dataloader_num_workers=4,
            load_best_model_at_end=True if eval_dataset else False
        )

        # Trainer initialization
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
            tokenizer=self.tokenizer,
        )

        # Start training
        logging.info("Starting training")
        trainer.train()

        # Save final model
        self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)
        logging.info(f"Model saved to {self.checkpoint_dir}")



trainer = Llama2Trainer(
    checkpoint_dir="./checkpoints",
    model_dir="/model-weights/Llama-2-7b-hf",
    epoch=3,  # Number of epochs
    batch=4,  # Batch size
    lr=3e-5,  # Learning rate
    fp16=True,  # Enable mixed precision training
)
# Load and prepare datasets
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the masked data
df = pd.read_csv("/h/emzed/data/qa_discharge_masked.csv")

# Split into train and eval sets (80-20 split)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to datasets
train_dataset = {
    'text': train_df['masked_text'].tolist(),
    'question': train_df['q'].tolist(),
    'answer': train_df['a'].tolist()
}

eval_dataset = {
    'text': eval_df['masked_text'].tolist(), 
    'question': eval_df['q'].tolist(),
    'answer': eval_df['a'].tolist()
}

trainer.train(train_dataset, eval_dataset)