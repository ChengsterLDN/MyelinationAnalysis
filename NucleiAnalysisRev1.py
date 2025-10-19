import cv2
import numpy as np
import os
import json
from tkinter import Tk, filedialog, messagebox


class NucleiAnalyser:
    def __init__(self, image_path, output_folder):
        self.image_path = image_path
        self.output_folder = output_folder
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.nuclei_count = []
        self.nuclei_prop = []

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

        for i, contour in enumerate(contours):
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
        
        # Save properties to JSON
        json_path = os.path.join(self.output_folder, "nuclei_properties.json")
        with open(json_path, 'w') as json_file:
            json.dump(self.nuclei_prop, json_file, indent=4)
        
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
        
        # Save visualisation
        vis_path = os.path.join(self.output_folder, "nuclei_count.png")
        cv2.imwrite(vis_path, vis_image)
        print(f"Visualisation saved to {vis_path}")

    def process(self):

        if not self.detect_nuclei():
            messagebox.showinfo("No Nuclei", "No nuclei of acceptable size were detected in the image.")
            return False
        
        self.visualise()
        return True
    
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(title="Select Nuclei Image")
    if not image_path:
        messagebox.showerror("Error", "Please select an image file.")
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
                print(f"Nuclei {prop['nuclei_id']}: Center({prop['x_c']}, {prop['y_c']}), "f"Area: {prop['area']:.2f}, Circularity: {prop['circularity']:.3f}")
        
        else:
            messagebox.showwarning("Warning", "Analysis completed with issues.")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    root.destroy()


            


