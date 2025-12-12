import os
import time
import json
import csv
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

    def load_pillar_coordinates(self, pillar_coords_path):
        """Load pillar coordinates from JSON file"""
        if os.path.exists(pillar_coords_path):
            with open(pillar_coords_path, 'r') as f:
                pillar_coords = json.load(f)
            print(f"Loaded {len(pillar_coords)} pillar coordinates from {pillar_coords_path}")
            return pillar_coords
        else:
            print(f"Warning: {pillar_coords_path} not found.")
            return []

    def load_nuclei_props(self, nuclei_props_path):
        """Load nuclei properties from JSON file and count nuclei"""
        if os.path.exists(nuclei_props_path):
            with open(nuclei_props_path, 'r') as f:
                nuclei_props = json.load(f)
            nuclei_count = len(nuclei_props)
            print(f"Loaded {nuclei_count} nuclei from {nuclei_props_path}")
            return nuclei_count, nuclei_props
        else:
            print(f"Warning: {nuclei_props_path} not found.")
            return 0, []

    def find_subfolders_with_boxes(self, parent_directory):
        """Find all subfolders that contain a 'boxes' folder and pillar_coords.json"""
        valid_subfolders = []
        for item in os.listdir(parent_directory):
            subfolder_path = os.path.join(parent_directory, item)
            if os.path.isdir(subfolder_path):
                boxes_path = os.path.join(subfolder_path, 'boxes')
                pillar_coords_path = os.path.join(subfolder_path, f'{item}_pillar_coords.json')
                nuclei_props_path = os.path.join(subfolder_path, 'nuclei_mip_nuclei_props.json')
                
                if os.path.exists(boxes_path) and os.path.isdir(boxes_path):
                    valid_subfolders.append({
                        'subfolder_path': subfolder_path,
                        'subfolder_name': item,
                        'boxes_path': boxes_path,
                        'pillar_coords_path': pillar_coords_path,
                        'nuclei_props_path': nuclei_props_path
                    })
        return valid_subfolders

    def predict_image(self, image_path):
        """Predict class for a single image"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.argmax().item()
    
    def process_boxes_folder(self, boxes_folder, pillar_coords):
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
                    
                    if pred_class == 3 and pillar_coords:
                        try:
                            box_num = int(img_file.split('_')[1].split('.')[0])
                            pillar_info = next((p for p in pillar_coords if p['cell_id'] == box_num), None)
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

    def save_summary_csv(self, summary_data, output_path):
        """Save summary data to CSV file"""
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['subfolder_name', 'nuclei_count', 'pillars_count', 
                         'class_0_count', 'class_1_count', 'class_2_count', 'class_3_count',
                         'processing_time_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data in summary_data:
                writer.writerow({
                    'subfolder_name': data['subfolder_name'],
                    'nuclei_count': data['nuclei_count'],
                    'pillars_count': data['pillars_count'],
                    'class_0_count': data['class_counts'][0],
                    'class_1_count': data['class_counts'][1],
                    'class_2_count': data['class_counts'][2],
                    'class_3_count': data['class_counts'][3],
                    'processing_time_seconds': data['processing_time']
                })
        
        print(f"Saved summary CSV to: {output_path}")

    def run_analysis(self):
        """Main method to run the complete analysis with user dialogs"""
        # Initialise Tkinter root
        root = Tk()
        root.withdraw()
        
        # Ask user for parent directory
        print("Please select the parent directory containing your subfolders...")
        parent_directory = filedialog.askdirectory(title="Select Parent Directory")
        if not parent_directory:
            messagebox.showerror("Error", "Please select a parent directory.")
            root.destroy()
            return
        
        # Find valid subfolders
        valid_subfolders = self.find_subfolders_with_boxes(parent_directory)
        if not valid_subfolders:
            messagebox.showinfo("Info", "No valid subfolders found. Each subfolder should contain a 'boxes' folder and pillar_coords.json.")
            root.destroy()
            return
        
        print(f"Found {len(valid_subfolders)} valid subfolders:")
        for subfolder_info in valid_subfolders:
            print(f"  - {subfolder_info['subfolder_name']}")
        
        # Process all subfolders
        total_start_time = time.time()
        all_wrapped_pillars = []
        total_class_3_count = 0
        summary_data = []
        
        for subfolder_info in valid_subfolders:
            print(f"\n=== Processing {subfolder_info['subfolder_name']} ===")
            
            # Load pillar coordinates for this subfolder
            pillar_coords = self.load_pillar_coordinates(subfolder_info['pillar_coords_path'])
            
            # Load nuclei count for this subfolder
            nuclei_count, nuclei_props = self.load_nuclei_props(subfolder_info['nuclei_props_path'])
            
            # Process the boxes folder
            result = self.process_boxes_folder(subfolder_info['boxes_path'], pillar_coords)
            
            # Print folder results
            print(f"\nPrediction Counts for {subfolder_info['subfolder_name']}:")
            print(f"  Nuclei count: {nuclei_count}")
            print(f"  Pillars count: {result['image_count']}")
            for class_id, count in result['class_counts'].items():
                print(f"  Class {class_id}: {count} images")
            print(f"  Total images processed: {result['image_count']}")
            print(f"  Time elapsed: {result['processing_time']:.2f} seconds")
            
            # Save wrapped pillars in the same subfolder
            if result['wrapped_pillars']:
                output_filename = f"{subfolder_info['subfolder_name']}_wrapped_pillars.json"
                output_path = os.path.join(subfolder_info['subfolder_path'], output_filename)
                self.save_wrapped_pillars(result['wrapped_pillars'], output_path)
                all_wrapped_pillars.extend(result['wrapped_pillars'])
                total_class_3_count += len(result['wrapped_pillars'])
                print(f"  Class 3 pillars found: {len(result['wrapped_pillars'])}")
            else:
                print(f"  No class 3 pillars found in this folder")
            
            # Add to summary data
            summary_data.append({
                'subfolder_name': subfolder_info['subfolder_name'],
                'nuclei_count': nuclei_count,
                'pillars_count': result['image_count'],
                'class_counts': result['class_counts'],
                'processing_time': result['processing_time']
            })
        
        # Save summary CSV
        csv_output_path = os.path.join(parent_directory, "analysis_summary.csv")
        self.save_summary_csv(summary_data, csv_output_path)
        
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        
        # Show final summary
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Total processing time: {total_elapsed_time:.2f} seconds")
        print(f"Total subfolders processed: {len(valid_subfolders)}")
        print(f"Total class 3 pillars found across all subfolders: {total_class_3_count}")
        print(f"Summary CSV saved to: {csv_output_path}")
        print(f"Individual results saved in respective subfolders")
        
        messagebox.showinfo(
            "Analysis Complete", 
            f"Analysis completed successfully!\n\n"
            f"Total subfolders processed: {len(valid_subfolders)}\n"
            f"Total class 3 pillars found: {total_class_3_count}\n"
            f"Summary CSV saved to: {csv_output_path}\n"
            f"Results saved in respective subfolders"
        )
        
        root.destroy()

if __name__ == "__main__":
    model_path = "./Run3New"

    try:
        analyser = MyelinScorer(model_path)
        analyser.run_analysis()
    except Exception as e:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during initialisation: {str(e)}")
        root.destroy()