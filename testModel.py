from PIL import Image
import requests
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# Load your saved model and processor
model = ViTForImageClassification.from_pretrained("./my_ring_completeness_model")
processor = ViTImageProcessor.from_pretrained("./my_ring_completeness_model")

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_ring_completeness(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Map index to label (adjust based on your training labels)
    id2label = {
        0: "non-existent",
        1: "less than half",
        2: "more than half",
        3: "complete"  
    }
    
    return id2label[predicted_class_idx], logits.softmax(dim=1)

# Test 
image_path = "C:\\Users\\jonat\\Myelination\\test_images\\cell_59.png"
predicted_class, probabilities = predict_ring_completeness(image_path)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")