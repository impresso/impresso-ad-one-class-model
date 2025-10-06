# fine_tune_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import datasets
import json

# Load your training data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    texts = [item['ft'] for item in data]
    labels = [0 if item['type'] == 'non-ad' else 1 for item in data]  # 0 = non-ad, 1 = ad
    return texts, labels

# Load dataset
train_texts, train_labels = load_data('ads_3000_finetuning.jsonl')
test_texts, test_labels = load_data('non_ads_3000_finetuning.jsonl')

# Tokenizer and model loading
tokenizer = AutoTokenizer.from_pretrained("classla/xlm-roberta-base-multilingual-text-genre-classifier")
model = AutoModelForSequenceClassification.from_pretrained("classla/xlm-roberta-base-multilingual-text-genre-classifier", num_labels=2)

# Tokenization of data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Prepare datasets
train_dataset = datasets.Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels})
test_dataset = datasets.Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': test_labels})

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tuning the model
trainer.train()
trainer.save_model('./fine_tuned_model')
