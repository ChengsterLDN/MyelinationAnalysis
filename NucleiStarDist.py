import cv2
import numpy as np
import os
import json
from tkinter import Tk, filedialog, messagebox
import glob


class NucleiAnalyser:
    def __init__(self, image_path, output_folder):
        self.image_path = image_path
        self.output_folder = output_folder
        self.image = None
        self.nuclei_count = []
        self.nuclei_prop = []

    def load_single_image(self):
        """Load a single image (original functionality)"""
        self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

    def preprocess(self):
        # Convert to greyscale and otsu threshold
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_cleaned = cv2.fastNlMeansDenoising(binary)
        
        return binary_cleaned
        
    
    def detect_nuclei(self):
        # Filter out nuclei of acceptable size
        binary_cleaned = self.preprocess()

        # Find them nuclei
        contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_CONTOUR_AREA = 20
        filtered_contours = []

        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                filtered_contours.append(cnt)

        final_binary = np.zeros_like(binary_cleaned)
        for cnt in filtered_contours:
            cv2.drawContours(final_binary, [cnt], -1, 255, -1)

        contours_final, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_SIZE_THRESHOLD = 450
        MAX_SIZE_THRESHOLD = 10000

        self.nuclei_count = []
        self.nuclei_prop = []

        for i, contour in enumerate(contours_final):
            area = cv2.contourArea(contour)

            # Calculate circularity 
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0 and area > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            # Filter by size and circularity 
            if (area >= MIN_SIZE_THRESHOLD and area <= MAX_SIZE_THRESHOLD 
                and circularity >= 0.01):
                
                self.nuclei_count.append(contour)
                
                # Calculate centroid 
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    xc = int(moments["m10"] / moments["m00"])
                    yc = int(moments["m01"] / moments["m00"])
                else:
                    xc, yc = 0, 0
                
                # Store properties
                properties = {
                    "nuclei_id": i,
                    "x_c": xc,
                    "y_c": yc,
                    "area": float(area),
                    "circularity": float(circularity)
                }
                self.nuclei_prop.append(properties)

        # Get parent folder name for naming
        if os.path.isdir(self.image_path):
            parent_folder_name = os.path.basename(os.path.normpath(self.image_path))
        else:
            parent_folder_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # Save properties to JSON
        json_path = os.path.join(self.output_folder, f"{parent_folder_name}_nuclei_props.json")
        with open(json_path, 'w') as json_file:
            json.dump(self.nuclei_prop, json_file, indent = 4)
        print(f"Detected {len(self.nuclei_count)} nuclei")
        print(f"Nuclei properties saved to {json_path}")
        
        return len(self.nuclei_count) > 0
        
    def visualise(self):
        if not self.nuclei_count:
            return
            
        vis_image = self.image.copy()

        for i, (contour, properties) in enumerate(zip(self.nuclei_count, self.nuclei_prop)):
            # Draw contour
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
                
            # Draw centroid
            xc = properties["x_c"]
            yc = properties["y_c"]
            cv2.circle(vis_image, (xc, yc), 5, (255, 0, 0), -1)
                
            # Add number label
            cv2.putText(vis_image, str(i), (xc + 10, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def process(self):
        # Load the nuclei_mip.png image directly
        self.load_single_image()

        if not self.detect_nuclei():
            messagebox.showinfo("No Nuclei", "No nuclei of acceptable size were detected in the image.")
            return False
        
        self.visualise()
        return True

def process_all_subfolders(parent_directory):
    """Process all subfolders in the parent directory that contain nuclei_mip.png"""
    processed_folders = 0
    successful_folders = 0
    
    # Get all subdirectories in the parent directory
    for folder_name in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Look for nuclei_mip.png in the subfolder
            nuclei_image_path = os.path.join(subfolder_path, "nuclei_mip.png")
            
            # Check if the nuclei image exists
            if os.path.exists(nuclei_image_path):
                print(f"Processing folder: {folder_name}")
                processed_folders += 1
                
                try:
                    # Process this folder - output goes to the same subfolder
                    analyser = NucleiAnalyser(nuclei_image_path, subfolder_path)
                    success = analyser.process()
                    
                    if success:
                        successful_folders += 1
                        print(f"✓ Successfully processed {folder_name} - Found {len(analyser.nuclei_count)} nuclei")
                    else:
                        print(f"✗ No nuclei detected in {folder_name}")
                        
                except Exception as e:
                    print(f"✗ Error processing {folder_name}: {str(e)}")
            else:
                print(f"Skipping {folder_name}: nuclei_mip.png not found")
    
    return processed_folders, successful_folders

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    # Ask for parent directory containing subfolders with nuclei_mip.png
    parent_directory = filedialog.askdirectory(title="Select Parent Directory Containing Subfolders")
    
    if not parent_directory:
        messagebox.showerror("Error", "Please select a parent directory.")
        exit()
    
    try:
        processed, successful = process_all_subfolders(parent_directory)
        
        messagebox.showinfo("Processing Complete", 
                           f"Processed {processed} folders\n"
                           f"Successfully completed: {successful}\n"
                           f"Failed: {processed - successful}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    root.destroy()