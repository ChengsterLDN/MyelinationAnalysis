import os
import time
import json
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from tkinter import Tk, filedialog, messagebox

class MyelinScorer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.pillar_coords = []
        self.boxes_folders = []

    def load_pillar_coordinates(self, pillar_coords_path):
        """Load pillar coordinates from JSON file"""
        if os.path.exists(pillar_coords_path):
            with open(pillar_coords_path, 'r') as f:
                self.pillar_coords = json.load(f)
            print(f"Loaded {len(self.pillar_coords)} pillar coordinates from {pillar_coords_path}")
            return True
        else:
            print(f"Warning: {pillar_coords_path} not found.")
            return False
        
    def find_boxes_folders(self, root_directory):
        """Recursively find all 'boxes' folders in the directory structure"""
        self.boxes_folders = []
        for root, dirs, files in os.walk(root_directory):
            if 'boxes' in dirs:
                self.boxes_folders.append(os.path.join(root, 'boxes'))
        return self.boxes_folders

    def predict_image(self, image_path):
        """Predict class for a single image"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.argmax().item()
    
    def process_boxes_folder(self, boxes_folder):
        """Process a single boxes folder and return class counts and class 3 pillars"""
        print(f"\nProcessing folder: {boxes_folder}")
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        wrapped_pillars = []
        
        folder_start_time = time.time()
        image_count = 0
        
        for img_file in os.listdir(boxes_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(boxes_folder, img_file)
                    pred_class = self.predict_image(img_path)
                    class_counts[pred_class] += 1
                    image_count += 1
                    
                    if pred_class == 3 and self.pillar_coords:
                        try:
                            box_num = int(img_file.split('_')[1].split('.')[0])
                            pillar_info = next((p for p in self.pillar_coords if p['cell_id'] == box_num), None)
                            if pillar_info:
                                wrapped_pillars.append({
                                    'cell_id': pillar_info['cell_id'],
                                    'image_filename': pillar_info['image_filename'],
                                    'center_coordinates': pillar_info['center_coordinates'],
                                    'predicted_class': 3
                                })
                        except (IndexError, ValueError):
                            print(f"  Could not parse box number from {img_file}")
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
        
        folder_end_time = time.time()
        folder_elapsed_time = folder_end_time - folder_start_time
        
        return {
            'class_counts': class_counts,
            'wrapped_pillars': wrapped_pillars,
            'image_count': image_count,
            'processing_time': folder_elapsed_time
        }
    
    def save_wrapped_pillars(self, wrapped_pillars, output_path):
        """Save class 3 pillars to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(wrapped_pillars, f, indent=2)
        print(f"Saved {len(wrapped_pillars)} class 3 pillar properties to: {output_path}")

    def run_analysis(self):
        """Main method to run the complete analysis with user dialogs"""
        # Initialize Tkinter root
        root = Tk()
        root.withdraw()
        
        # Ask user for root directory
        print("Please select the root directory containing your data...")
        root_directory = filedialog.askdirectory(title="Select Root Directory")
        if not root_directory:
            messagebox.showerror("Error", "Please select a root directory.")
            root.destroy()
            return
        
        # Ask user for pillar coordinates file
        print("Please select the pillar coordinates JSON file...")
        pillar_coords_path = filedialog.askopenfilename(
            title="Select Pillar Coordinates JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not pillar_coords_path:
            messagebox.showerror("Error", "Please select a pillar coordinates JSON file.")
            root.destroy()
            return
        
        # Ask user for output directory
        print("Please select the output directory...")
        output_directory = filedialog.askdirectory(title="Select Output Directory")
        if not output_directory:
            messagebox.showerror("Error", "Please select an output directory.")
            root.destroy()
            return
        
        # Load pillar coordinates
        if not self.load_pillar_coordinates(pillar_coords_path):
            messagebox.showwarning("Warning", "Pillar coordinates file not found. Continuing without coordinate filtering.")
        
        # Find boxes folders
        boxes_folders = self.find_boxes_folders(root_directory)
        if not boxes_folders:
            messagebox.showinfo("Info", "No 'boxes' folders found in the selected directory.")
            root.destroy()
            return
        
        print(f"Found {len(boxes_folders)} boxes folders:")
        for folder in boxes_folders:
            print(f"  - {folder}")
        
        # Process all folders
        total_start_time = time.time()
        all_wrapped_pillars = []
        
        for boxes_folder in boxes_folders:
            result = self.process_boxes_folder(boxes_folder)
            
            # Print folder results
            parent_folder_name = os.path.basename(os.path.dirname(boxes_folder))
            print(f"\nPrediction Counts for {parent_folder_name}/boxes:")
            for class_id, count in result['class_counts'].items():
                print(f"  Class {class_id}: {count} images")
            print(f"  Total images processed: {result['image_count']}")
            print(f"  Time elapsed: {result['processing_time']:.2f} seconds")
            
            # Save individual folder results
            if result['wrapped_pillars']:
                output_filename = f"wrapped_pillars_{parent_folder_name}.json"
                output_path = os.path.join(output_directory, output_filename)
                self.save_wrapped_pillars(result['wrapped_pillars'], output_path)
                all_wrapped_pillars.extend(result['wrapped_pillars'])
            else:
                print(f"  No class 3 pillars found in this folder")
        
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        
        # Save combined results
        if all_wrapped_pillars:
            combined_output_path = os.path.join(output_directory, "all_wrapped_pillars.json")
            self.save_wrapped_pillars(all_wrapped_pillars, combined_output_path)
        
        # Show final summary
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Total processing time: {total_elapsed_time:.2f} seconds")
        print(f"Total class 3 pillars found: {len(all_wrapped_pillars)}")
        print(f"Results saved to: {output_directory}")
        
        messagebox.showinfo(
            "Analysis Complete", 
            f"Analysis completed successfully!\n\n"
            f"Total class 3 pillars found: {len(all_wrapped_pillars)}\n"
            f"Results saved to: {output_directory}"
        )
        
if __name__ == "__main__":

    model_path = "./Modelv1.4/Run3New"

    try:
        analyser = MyelinScorer(model_path)
        analyser.run_analysis()
    except Exception as e:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during initialization: {str(e)}")
        root.destroy()