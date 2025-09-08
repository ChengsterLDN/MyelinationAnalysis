from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset

import numpy as np
import evaluate

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imagefolder", data_dir = "C:\\Users\\jonat\\Myelination\\dataset")

# Load Pretrained ViT model and processor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("./Modelv1.4/Run2",
                                                num_labels=4,  # 0, 1, 2, 3
                                                ignore_mismatched_sizes=True)


# Move model to GPU
model = model.to(device)

# Preprocess function
def preprocess(examples):

    # Process images
    inputs = processor(images=examples["image"], return_tensors="pt")
    
    # Add labels if they exist
    if "label" in examples:
        inputs["labels"] = examples["label"]
    
    return inputs

    #return processor(images=examples["image"], return_tensors="pt")

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
    output_dir="./Modelv1.4",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=50,  # Add logging to see progress
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

#print(torch.cuda.memory_allocated(device)/1024**2, "MB")
#print(torch.cuda.memory_reserved(device)/1024**2, "MB")


