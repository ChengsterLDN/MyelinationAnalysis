import os
import time
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# 1. Load your trained model
model_path = "./Modelv1.1/checkpoint-561"  
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Define prediction function
def predict_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.argmax().item()

# 3. Process all images in folder
test_folder = "./test_images"  
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

start_time = time.time()
for img_file in os.listdir(test_folder):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img_path = os.path.join(test_folder, img_file)
            pred_class = predict_image(img_path)
            class_counts[pred_class] += 1
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

end_time = time.time()
elapsed_time = end_time - start_time

# 4. Print results
print("\nPrediction Counts:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} images")
print(f"Time elapsed: {elapsed_time:.2f} seconds")