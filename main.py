from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset

import numpy as np
import evaluate

dataset = load_dataset("imagefolder", data_dir = "C:\\Users\\Jonathon Cheng\\Myelination\\dataset")

# Load Pretrained ViT model and processor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",
                                                num_labels=4,  # 0, 1, 2, 3
                                                ignore_mismatched_sizes=True)

# Preprocess function
def preprocess(examples):
    return processor(images=examples["image"], return_tensors="pt")

# Preprocessing 

dataset = dataset.map(preprocess, batched = True)

# Evaluation

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references = labels)

# Training setup (simplified)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
