import os
import time
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification


model_path = "./Modelv1.4/Run2"  
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def predict_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.argmax().item()


root_directory = 'C:/Users/jonat/Documents/My Documents/MecBioMed/MyelinationProject/MBP DATA/MBP V5 coating/Fibronectin'
#root_directory = './test_images' 

boxes_folders = []
for root, dirs, files in os.walk(root_directory):
    if 'boxes' in dirs:
        boxes_folders.append(os.path.join(root, 'boxes'))

print(f"Found {len(boxes_folders)} boxes folders:")
for folder in boxes_folders:
    print(f"  - {folder}")


total_start_time = time.time()

for boxes_folder in boxes_folders:
    print(f"\nProcessing folder: {boxes_folder}")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    folder_start_time = time.time()
    
    image_count = 0
    for img_file in os.listdir(boxes_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(boxes_folder, img_file)
                pred_class = predict_image(img_path)
                class_counts[pred_class] += 1
                image_count += 1
                #print(f"  {img_file}: Class {pred_class}")
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
    
    folder_end_time = time.time()
    folder_elapsed_time = folder_end_time - folder_start_time
    
    print(f"Prediction Counts for {os.path.basename(os.path.dirname(boxes_folder))}/boxes:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} images")
    print(f"  Total images processed: {image_count}")
    print(f"  Time elapsed: {folder_elapsed_time:.2f} seconds")

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

print(f"\nTotal processing time for all folders: {total_elapsed_time:.2f} seconds")