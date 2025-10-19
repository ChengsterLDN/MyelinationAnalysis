import cv2
import os
import numpy as np
import json
from tkinter import Tk, filedialog, messagebox
from scipy.ndimage import convolve

#os.makedirs('boxes', exist_ok=True)


class AutoBoxer:

    def __init__(self, pillar_image_path, myelin_image_path, output_folder):
        self.pillar_image_path = pillar_image_path        
        self.myelin_image_path = myelin_image_path
        
        self.output_folder = output_folder
        self.dot_positions = []
        self.box_positions = []  

        self.pillar_image = cv2.imread(pillar_image_path, cv2.IMREAD_COLOR)
        self.myelin_image = cv2.imread(myelin_image_path, cv2.IMREAD_COLOR)

        # Check if images loaded successfully
        if self.pillar_image is None:
            raise ValueError(f"Failed to load pillar image: {pillar_image_path}")
        if self.myelin_image is None:
            raise ValueError(f"Failed to load myelin image: {myelin_image_path}")
        
        # Calculate scales for coordinate conversion
        self.display_pillar_scale = 1.0  
        self.overlay_scale_x = 1.0
        self.overlay_scale_y = 1.0
        
    def extract(self):

        """Detect centers of pillars"""

        gray_image = cv2.cvtColor(self.pillar_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        radius = 15
        y, x = np.ogrid[-radius: radius+1, -radius: radius+1]
        circular_filter = x**2 + y**2 <= radius**2
        filtered_image = convolve(binary_mask, circular_filter.astype(float))
        contours, _ = cv2.findContours(filtered_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.dot_positions = []
        self.dot_visuals = []

        min_area = 2250
        img_h, img_w = binary_mask.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:  
                continue
            moments = cv2.moments(contour)

            # Check if contour touches image border
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 >= img_h:
                continue  # Skip partial pillars touching edges

            if moments["m00"] != 0:
                xc = int(moments["m10"] / moments["m00"])
                yc = int(moments["m01"] / moments["m00"])
                self.dot_positions.append((xc, yc))

        if not self.dot_positions:
            messagebox.showinfo("Detection Complete", "No pillars detected.")
            return False   
        return True

    def create_box(self):
        """Create boxes around detected centers and display scoring buttons."""
        box_size = 100
        self.box_positions = []  # Reset box positions

        # Sort dot positions by top-to-bottom, left-to-right
        self.dot_positions = sorted(self.dot_positions, key=lambda pos: (pos[1], pos[0]))

        for (x, y) in self.dot_positions:
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(self.myelin_image.shape[1], x + box_size // 2)
            y2 = min(self.myelin_image.shape[0], y + box_size // 2)

            # Store the rectangle's position
            self.box_positions.append((x1, y1, x2, y2))

        print(f"Created {len(self.box_positions)} boxes.")
        return True
    
    def save_box(self):

        """Save all cropped box images."""

        if not self.box_positions:
            messagebox.showwarning("No Boxes", "Please create boxes before saving.")
            return

        # Load the original myelin image for cropping
        base_image = cv2.imread(self.myelin_image_path, cv2.IMREAD_COLOR)
        
        cell_data = []

        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            # Add padding and ensure bounds are within the image dimensions
            y1_cropped = max(0, y1)
            y2_cropped = min(base_image.shape[0], y2)
            x1_cropped = max(0, x1)
            x2_cropped = min(base_image.shape[1], x2)
            cropped_img = base_image[y1_cropped:y2_cropped, x1_cropped:x2_cropped]
            file_path = os.path.join(self.output_folder, "boxes", f"box_{i}.png")
            cv2.imwrite(file_path, cropped_img)

            xc, yc = self.dot_positions[i]

             # Store data for this pillar
            cell_info = {
                "cell_id": i,
                "image_filename": f"box_{i}.png",
                "center_coordinates": {
                    "x": xc,
                    "y": yc
                }
            }

            cell_data.append(cell_info)
            print(f"Saved {file_path}")
        
        json_path = os.path.join(self.output_folder, "pillar_coords.json")
        with open(json_path, 'w') as json_file:
            json.dump(cell_data, json_file, indent=4)
        print(f"All {len(self.box_positions)} boxes have been saved.")
        print(f"Pillar coordinates saved to {json_path}")
        return True
        
    def process(self):
        """Main processing pipeline."""
        if not self.extract():
            return False
        
        if not self.create_box():
            return False
            
        if not self.save_box():
            return False
        return True

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window
    pillar_image_path = filedialog.askopenfilename(title="Select Pillar Image")
    myelin_image_path = filedialog.askopenfilename(title="Select Myelin Image")
    output_folder = filedialog.askdirectory(title="Select Output Directory")

    if pillar_image_path and myelin_image_path and output_folder:
        os.makedirs(os.path.join(output_folder, "boxes"), exist_ok=True)
        
        analyser = AutoBoxer(pillar_image_path, myelin_image_path, output_folder)
        success = analyser.process()

        if success:
            messagebox.showinfo("Success", "Processing completed successfully!")
        else:
            messagebox.showwarning("Warning", "Processing completed with issues.")
    else:
        messagebox.showerror("Error", "Please select both images and an output directory.")
    
    root.destroy()