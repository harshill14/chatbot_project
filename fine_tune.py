from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from datasets import Dataset

# Load dataset
def load_dataset():
    data = pd.read_csv('data/conversations.csv')
    dataset = Dataset.from_pandas(data)
    return dataset


model_name = "gpt3.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


dataset = load_dataset()
tokenized_dataset = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir='./models/fine_tuned_model',
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)


trainer.train()
trainer.save_model("./models/fine_tuned_model")
tokenizer.save_pretrained("./models/fine_tuned_model")
