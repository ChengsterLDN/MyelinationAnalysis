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

    def process_folder(self, input_folder):
  
        print(f"\nProcessing folder: {input_folder}")
        
        # Create output directories
        valid_dir = os.path.join(input_folder, "valid")
        invalid_dir = os.path.join(input_folder, "invalid")
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(invalid_dir, exist_ok=True)
        
        # Get all PNG files
        png_files = [f for f in os.listdir(input_folder) 
                    if f.lower().endswith('.png') and os.path.isfile(os.path.join(input_folder, f))]
        
        if not png_files:
            print("No PNG files found in the selected folder.")
            return {"valid": 0, "invalid": 0, "total": 0}
        
        print(f"Found {len(png_files)} PNG files to process")
        
        # Process images
        results = {"valid": 0, "invalid": 0, "total": len(png_files)}
        processed_count = 0
        
        start_time = time.time()
        
        for img_file in png_files:
            try:
                img_path = os.path.join(input_folder, img_file)
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
                    
                    print(f"  {img_file}: {class_name} (confidence: {confidence:.3f})")
                else:
                    # If prediction failed, move to invalid folder
                    destination = os.path.join(invalid_dir, img_file)
                    shutil.move(img_path, destination)
                    results["invalid"] += 1
                    print(f"  {img_file}: prediction failed - moved to invalid")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                # Move problematic files to invalid folder
                try:
                    destination = os.path.join(invalid_dir, img_file)
                    shutil.move(os.path.join(input_folder, img_file), destination)
                    results["invalid"] += 1
                except:
                    pass
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nProcessing complete!")
        print(f"Valid images: {results['valid']}")
        print(f"Invalid images: {results['invalid']}")
        print(f"Total processed: {processed_count}/{len(png_files)}")
        print(f"Time elapsed: {processing_time:.2f} seconds")
        print(f"Valid images moved to: {valid_dir}")
        print(f"Invalid images moved to: {invalid_dir}")
        
        return results

    def run_validation(self):
        root = Tk()
        root.withdraw()
        
        # Ask user for input folder
        print("Please select the folder containing PNG images to validate...")
        input_folder = filedialog.askdirectory(title="Select Folder with PNG Images")
        if not input_folder:
            messagebox.showerror("Error", "Please select a folder containing PNG images.")
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
            results = self.process_folder(input_folder)
            
            # Show results summary
            messagebox.showinfo(
                "Validation Complete", 
                f"Image validation completed!\n\n"
                f"Valid images: {results['valid']}\n"
                f"Invalid images: {results['invalid']}\n"
                f"Total processed: {results['total']}\n\n"
                f"Images have been moved into 'valid' and 'invalid' folders."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during validation: {str(e)}")
        
        finally:
            root.destroy()

if __name__ == "__main__":
    try:
        # Initialize without a model path - user must select one
        validator = StackValidator(None)
        validator.run_validation()

    except Exception as e:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during initialisation: {str(e)}")
        root.destroy()