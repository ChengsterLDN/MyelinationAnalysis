import cv2
import os
import numpy as np
from tkinter import Tk, Button, Canvas, Label, Toplevel, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
from scipy.ndimage import convolve

#os.makedirs('boxes', exist_ok=True)


class MyelinAnalyzerApp:

    def __init__(self, root, pillar_image_path, myelin_image_path):
        self.root = root
        self.root.title("Myelin Analyzer")
        # Store image paths for later saving
        self.pillar_image_path = pillar_image_path        
        self.myelin_image_path = myelin_image_path
        self.dot_positions = []
        self.box_positions = []  
        self.coordinate_matrix = [] 
        self.current_cell_index = 0  
        self.canvas = Canvas(root)
        self.myelin_canvas = Canvas(root)
        self.pillar_image = cv2.imread(pillar_image_path, cv2.IMREAD_COLOR)
        self.myelin_image = cv2.imread(myelin_image_path, cv2.IMREAD_COLOR)
        self.display_myelin_image = self.resize_image(self.myelin_image)
        self.myelin_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_myelin_image, cv2.COLOR_BGR2RGB)))
        self.myelin_canvas.create_image(0, 0, anchor='nw', image=self.myelin_photo)
        self.is_select_empty_mode = False
        self.empty_cells = set()

# Resize images to fit the window while keeping aspect ratio

        self.display_pillar_image = self.resize_image(self.pillar_image)
        self.display_myelin_image = self.resize_image(self.myelin_image)
        self.display_pillar_scale = self.pillar_image.shape[1] / self.display_pillar_image.shape[1]  # Define display scale
        self.canvas = Canvas(root, width=self.display_pillar_image.shape[1], height=self.display_pillar_image.shape[0])
        self.canvas.grid(row=0, column=0)

        # Create a canvas to allow grid drawing and dragging

# Create a canvas for displaying the overlay image beside the pillar image
        self.myelin_canvas = Canvas(root, width=self.display_myelin_image.shape[1], height=self.display_myelin_image.shape[0])
        self.myelin_canvas.grid(row=0, column=1)
        self.myelin_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_myelin_image, cv2.COLOR_BGR2RGB)))
        self.myelin_canvas.create_image(0, 0, anchor='nw', image=self.myelin_photo)
        self.pillar_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_pillar_image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.pillar_photo)
        self.overlay_scale_x = self.myelin_image.shape[1] / self.display_myelin_image.shape[1]
        self.overlay_scale_y = self.myelin_image.shape[0] / self.display_myelin_image.shape[0]
        self.extract()

    def overlay_pillar_on_myelin(self, alpha=0.3):

        """Overlay the pillar image on the myelin image, save it, and update display."""
        if self.pillar_image.shape[:2] != self.myelin_image.shape[:2]:
            messagebox.showwarning("Warning", "Pillar and myelin images have different dimensions. Resizing may distort the images.")
        # Resize the pillar image to match the myelin image dimensions
        pillar_resized = cv2.resize(self.pillar_image, (self.myelin_image.shape[1], self.myelin_image.shape[0]))

        # Convert images from BGR to RGB for correct color representation in PIL
        myelin_rgb = cv2.cvtColor(self.myelin_image, cv2.COLOR_BGR2RGB)
        pillar_rgb = cv2.cvtColor(pillar_resized, cv2.COLOR_BGR2RGB)

        # Convert images to RGBA format to support transparency in PIL
        myelin_rgba = Image.fromarray(myelin_rgb).convert("RGBA")
        pillar_rgba = Image.fromarray(pillar_rgb).convert("RGBA")

        # Set the alpha channel for the pillar image to control transparency
        pillar_rgba.putalpha(int(255 * alpha))

        # Composite the translucent pillar image on top of the myelin image
        blended_image = Image.alpha_composite(myelin_rgba, pillar_rgba)

        # Save the overlay image
        overlay_image_path = "overlay_myelin_pillar.png"
        blended_image.save(overlay_image_path)
        self.overlay_image_path = overlay_image_path  # Store the path for later reference

        # Convert back to BGR format for OpenCV and update the myelin image
        self.myelin_image = cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGBA2BGR)

    def resize_image(self, image, max_size=600):
        h, w = image.shape[:2]
        if h <= max_size and w <= max_size:  # Add this check
            return image  # Return the original image if it's already small enough
        scale = min(max_size / h, max_size / w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def extract(self):

        """Detect centers of pillars and mark them as editable dots."""

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
            if area < min_area:  # Skip small areas (cut-off pillars, noise)
                continue

            moments = cv2.moments(contour)

            # Check if contour touches image border
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 >= img_h:
                continue  # Skip partial pillars touching edges

            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                display_x, display_y = int(center_x / self.display_pillar_scale), int(center_y / self.display_pillar_scale)
                self.dot_positions.append((display_x, display_y))
                self.dot_visuals.append(self.canvas.create_oval(display_x - 2, display_y - 2, display_x + 2, display_y + 2, fill="red"))

        if not self.dot_positions:
            messagebox.showinfo("Detection Complete", "No pillars detected.")

        #else:
            #messagebox.showinfo("Detection Complete", f"Detected {len(self.dot_positions)} pillars.")

        """Create boxes around detected centers and display scoring buttons."""
        box_size = 30
        self.canvas.delete("box")
        self.myelin_canvas.delete("box")

        self.box_positions = []  # Reset box positions

        # Sort dot positions by top-to-bottom, left-to-right
        self.dot_positions = sorted(self.dot_positions, key=lambda pos: (pos[1], pos[0]))

        for (x, y) in self.dot_positions:
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(self.display_myelin_image.shape[1], x + box_size // 2)
            y2 = min(self.display_myelin_image.shape[0], y + box_size // 2)

            # Store the rectangle's position
            self.box_positions.append((x1, y1, x2, y2))

            # Draw a blue rectangle on the canvas
            self.myelin_canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, tags="box")

        #print(f"Created {len(self.box_positions)} boxes.")


        """Save all cropped box images."""

        if not self.box_positions:
            messagebox.showwarning("No Boxes", "Please create boxes before saving.")
            return

        # Load the original myelin image for cropping
        base_image = cv2.imread(self.myelin_image_path, cv2.IMREAD_COLOR)

        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            # Scale coordinates back to the original image size
            x1_orig = int(x1 * self.overlay_scale_x)
            y1_orig = int(y1 * self.overlay_scale_y)
            x2_orig = int(x2 * self.overlay_scale_x)
            y2_orig = int(y2 * self.overlay_scale_y)

            # Add padding and ensure bounds are within the image dimensions
            y1_cropped = max(0, y1_orig)
            y2_cropped = min(base_image.shape[0], y2_orig)
            x1_cropped = max(0, x1_orig)
            x2_cropped = min(base_image.shape[1], x2_orig)

            cropped_img = base_image[y1_cropped:y2_cropped, x1_cropped:x2_cropped]
            file_path = output_folder + f"/boxes/box_{i}.png"
            cv2.imwrite(file_path, cropped_img)
            print(f"Saved {file_path}")
        
        #messagebox.showinfo("Save Complete", f"All {len(self.box_positions)} boxes have been saved to 'boxes' folder.")
        root.destroy()



    def get_cell_bounds(self, row, col):

        """Get the bounding box for a cell based on current grid line positions."""

        x1 = int(self.vertical_lines[col])
        y1 = int(self.horizontal_lines[row])
        x2 = int(self.vertical_lines[col + 1])
        y2 = int(self.horizontal_lines[row + 1])
        return x1, y1, x2, y2

if __name__ == "__main__":
    root = Tk()
    pillar_image_path = filedialog.askopenfilename(title="Select Pillar Image")
    myelin_image_path = filedialog.askopenfilename(title="Select Myelin Image")
    output_folder = filedialog.askdirectory(title="Select Output Directory")
    os.makedirs(output_folder + f'/boxes', exist_ok=True)

    

    if pillar_image_path and myelin_image_path:
        app = MyelinAnalyzerApp(root, pillar_image_path, myelin_image_path)
        root.mainloop()

    if not pillar_image_path or not myelin_image_path:
        messagebox.showerror("Error", "Please select both pillar and myelin images.")
        root.destroy()
