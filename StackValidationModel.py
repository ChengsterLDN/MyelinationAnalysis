from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset

import numpy as np
import evaluate

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imagefolder", data_dir = "C:\\Users\\jonat\\Documents\\GitHub\\MyelinationAnalysis\\MBPValidationDataset")

# Load Pretrained ViT model and processor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("./StackValidationv1.0/MBPRun",
                                                num_labels=2,  # valid, invalid
                                                ignore_mismatched_sizes=True)


# Move model to GPU
model = model.to(device)


def preprocess(examples):

    inputs = processor(images=examples["image"], return_tensors="pt")

    if "label" in examples:
        inputs["labels"] = examples["label"]
    
    return inputs


# Preprocessing 

dataset = dataset.map(preprocess, batched=True, remove_columns=["image"])

# Evaluation

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references = labels)

# Training setup (simplified)
training_args = TrainingArguments(
    output_dir="./StackValidationv1.0",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    num_train_epochs=3,
    save_steps=250,
    eval_steps=250,
    logging_dir="./logs",
    logging_steps=25,  # Add logging to see progress
    report_to="none",   # Disable wandb/etc if not needed
    #load_best_model_at_end=True
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
#trainer.train(resume_from_checkpoint=True)



