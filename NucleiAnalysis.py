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

    def mip(self, folder_path):

        """Create Maximum Intensity Projection from Z-stack images"""
        # Look for PNG files in the nuclei subfolders
        image_files = glob.glob(os.path.join(folder_path, "**", "nuclei", "*.png"), recursive=True)
        
        if not image_files:
            raise ValueError(f"No PNG files found in nuclei subfolders of {folder_path}")
        
        print(f"Found {len(image_files)} images for MIP creation")
        
        # Get parent folder name for naming
        parent_folder_name = os.path.basename(os.path.normpath(folder_path))

        
        # Read all images and stack them
        z_stack = []
        for img_file in sorted(image_files):
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                z_stack.append(img)
            else:
                print(f"Warning: Could not load {os.path.basename(img_file)}")
        
        if not z_stack:
            raise ValueError("No valid images could be loaded for MIP creation")
        
        # Create Maximum Intensity Projection
        mip = np.max(z_stack, axis=0)
        
        # Convert to 8-bit
        mip = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Save MIP
        mip_path = os.path.join(self.output_folder, f"{parent_folder_name}_mip.png")
        cv2.imwrite(mip_path, mip)
        print(f"MIP saved to {mip_path}")
        
        # Convert to 3-channel for compatibility with existing code
        self.image = cv2.cvtColor(mip, cv2.COLOR_GRAY2BGR)
        
        return mip_path

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
        # Check if input is a folder (Z-stack) or single image
        if os.path.isdir(self.image_path):
            print("Z-stack folder detected, creating MIP...")
            self.mip(self.image_path)
        else:
            print("Single image detected, loading directly...")
            self.load_single_image()

        if not self.detect_nuclei():
            messagebox.showinfo("No Nuclei", "No nuclei of acceptable size were detected in the image.")
            return False
        
        self.visualise()
        return True
    
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    # Ask user if they want to analyze a single image or a Z-stack folder
    choice = messagebox.askquestion("Input Type", 
                                   "Do you want to analyse a single image?\n\n"
                                   "Click 'Yes' for single image\n"
                                   "Click 'No' for Z-stack folder")
    
    if choice == 'yes':
        image_path = filedialog.askopenfilename(
            title="Select Nuclei Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
        )
    else:
        image_path = filedialog.askdirectory(title="Select Z-stack Folder")
    
    if not image_path:
        messagebox.showerror("Error", "Please select an image file or folder.")
        exit()
    
    output_folder = filedialog.askdirectory(title="Select Output Directory")
    if not output_folder:
        messagebox.showerror("Error", "Please select an output directory.")
        exit()
    
    try:
        analyser = NucleiAnalyser(image_path, output_folder)
        success = analyser.process()
        
        if success:
            messagebox.showinfo("Success", f"Analysis completed! Found {len(analyser.nuclei_count)} large nuclei.")
        
            # Print summary of stored properties
            print("\n=== STORED NUCLEI PROPERTIES ===")
            for prop in analyser.nuclei_prop:
                print(f"Nuclei {prop['nuclei_id']}: Centre({prop['x_c']}, {prop['y_c']}), "f"Area: {prop['area']:.2f}, Circularity: {prop['circularity']:.3f}")
        
        else:
            messagebox.showwarning("Warning", "Analysis completed with issues.")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    root.destroy()