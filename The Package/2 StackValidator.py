import os
import time
import shutil
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from tkinter import Tk, filedialog, messagebox

class StackValidator:
    def __init__(self, model_path):
        self.model_path = model_path
        # Don't load model here - wait for user selection
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define class labels - adjust these based on your model's classes
        self.class_labels = {
            0: "invalid",
            1: "valid"
        }

    def predict_image(self, image_path):
        """Predict class for a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_class = outputs.logits.argmax().item()
            confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][predicted_class].item()
            return predicted_class, confidence
        except Exception as e:
            print(f"Error predicting image {image_path}: {str(e)}")
            return None, 0.0

    def process_subfolder(self, subfolder_path, folder_type):
        """Process a single subfolder (mbp or pillar)"""
        print(f"  Processing {folder_type} folder: {subfolder_path}")
        
        # Create output directories within the subfolder
        valid_dir = os.path.join(subfolder_path, "valid")
        invalid_dir = os.path.join(subfolder_path, "invalid")
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(invalid_dir, exist_ok=True)
        
        # Get all PNG files in the subfolder
        png_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.png') and os.path.isfile(os.path.join(subfolder_path, f))]
        
        if not png_files:
            print(f"    No PNG files found in {folder_type} folder.")
            return {"valid": 0, "invalid": 0, "total": 0}
        
        print(f"    Found {len(png_files)} PNG files to process")
        
        # Process images
        results = {"valid": 0, "invalid": 0, "total": len(png_files)}
        processed_count = 0
        
        for img_file in png_files:
            try:
                img_path = os.path.join(subfolder_path, img_file)
                predicted_class, confidence = self.predict_image(img_path)
                
                if predicted_class is not None:
                    class_name = self.class_labels.get(predicted_class, "unknown")
                    
                    # Move image to appropriate folder
                    if class_name == "valid":
                        destination = os.path.join(valid_dir, img_file)
                        results["valid"] += 1
                    else:  # invalid or unknown
                        destination = os.path.join(invalid_dir, img_file)
                        results["invalid"] += 1
                    
                    shutil.move(img_path, destination)
                    processed_count += 1
                    
                    print(f"      {img_file}: {class_name} (confidence: {confidence:.3f})")
                else:
                    # If prediction failed, move to invalid folder
                    destination = os.path.join(invalid_dir, img_file)
                    shutil.move(img_path, destination)
                    results["invalid"] += 1
                    print(f"      {img_file}: prediction failed - moved to invalid")
                    
            except Exception as e:
                print(f"    Error processing {img_file}: {str(e)}")
                # Move problematic files to invalid folder
                try:
                    destination = os.path.join(invalid_dir, img_file)
                    shutil.move(os.path.join(subfolder_path, img_file), destination)
                    results["invalid"] += 1
                except:
                    pass
        
        print(f"    {folder_type} folder complete: {results['valid']} valid, {results['invalid']} invalid")
        return results

    def process_root_folder(self, parent_folder):
        """Process the parent folder and find all mbp and pillar subfolders recursively"""
        print(f"\nScanning parent folder: {parent_folder}")
        
        # Find all mbp and pillar folders recursively
        target_folders = []
        for root, dirs, files in os.walk(parent_folder):
            for dir_name in dirs:
                if dir_name.lower() in ["mbp", "pillar"]:
                    folder_path = os.path.join(root, dir_name)
                    target_folders.append((folder_path, dir_name))
        
        if not target_folders:
            print("No 'mbp' or 'pillar' folders found in the selected directory or its subdirectories.")
            return {"valid": 0, "invalid": 0, "total": 0, "folders_processed": 0}
        
        print(f"Found {len(target_folders)} target folders:")
        for folder_path, folder_name in target_folders:
            print(f"  - {folder_name}: {folder_path}")
        
        # Process each target folder
        total_results = {"valid": 0, "invalid": 0, "total": 0, "folders_processed": len(target_folders)}
        folder_details = {}
        
        start_time = time.time()
        
        for folder_path, folder_name in target_folders:
            print(f"\n--- Processing {folder_name} at: {os.path.relpath(folder_path, parent_folder)} ---")
            folder_results = self.process_subfolder(folder_path, folder_name)
            
            # Accumulate totals
            total_results["valid"] += folder_results["valid"]
            total_results["invalid"] += folder_results["invalid"]
            total_results["total"] += folder_results["total"]
            
            # Store folder details with relative path for clarity
            rel_path = os.path.relpath(folder_path, parent_folder)
            folder_key = f"{folder_name} ({rel_path})"
            folder_details[folder_key] = folder_results
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print summary
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Parent folder: {parent_folder}")
        print(f"Target folders processed: {total_results['folders_processed']}")
        print(f"Total valid images: {total_results['valid']}")
        print(f"Total invalid images: {total_results['invalid']}")
        print(f"Total images processed: {total_results['total']}")
        print(f"Time elapsed: {processing_time:.2f} seconds")
        
        # Print per-folder breakdown
        print(f"\nFolder breakdown:")
        for folder_name, results in folder_details.items():
            print(f"  {folder_name}: {results['valid']} valid, {results['invalid']} invalid")
        
        return total_results

    def run_validation(self):
        root = Tk()
        root.withdraw()
        
        # Ask user for parent folder
        print("Please select the parent folder containing 'mbp' and 'pillar' subfolders...")
        parent_folder = filedialog.askdirectory(title="Select Parent Folder with mbp and pillar subfolders")
        if not parent_folder:
            messagebox.showerror("Error", "Please select a parent folder containing 'mbp' and 'pillar' subfolders.")
            root.destroy()
            return
        
        # Ask user for model path (required - no default)
        print("Please select the model directory...")
        model_path = filedialog.askdirectory(title="Select Model Directory")

        if not model_path:
            messagebox.showerror("Error", "Model directory selection is required.")
            root.destroy()
            return

        # Load the model from user-selected path
        try:
            print(f"Loading model from: {model_path}")
            self.model = ViTForImageClassification.from_pretrained(model_path)
            self.processor = ViTImageProcessor.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model_path = model_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model from {model_path}: {str(e)}")
            root.destroy()
            return
        
        # Run validation
        try:
            results = self.process_root_folder(parent_folder)
            
            # Show results summary
            messagebox.showinfo(
                "Validation Complete", 
                f"Image validation completed!\n\n"
                f"Parent folder processed: {os.path.basename(parent_folder)}\n"
                f"Target folders processed: {results['folders_processed']}\n"
                f"Total valid images: {results['valid']}\n"
                f"Total invalid images: {results['invalid']}\n"
                f"Total images processed: {results['total']}\n\n"
                f"Images have been moved into 'valid' and 'invalid' subfolders\n"
                f"within each found 'mbp' and 'pillar' folder."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during validation: {str(e)}")
        
        finally:
            root.destroy()

if __name__ == "__main__":
    try:
        validator = StackValidator(None)
        validator.run_validation()

    except Exception as e:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during initialisation: {str(e)}")
        root.destroy()