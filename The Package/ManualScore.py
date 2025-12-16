import cv2
import os
import numpy as np
from tkinter import Tk, Button, Canvas, Label, Toplevel, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
from scipy.ndimage import convolve
import sys  # Add this to imports
import argparse  # Add this to imports

# Set up directories for saving

os.makedirs('0', exist_ok=True)
os.makedirs('1', exist_ok=True)
os.makedirs('2', exist_ok=True)
os.makedirs('3', exist_ok=True)

class MyelinAnalyser:

    def __init__(self, root, pillar_image_path, myelin_image_path):
        self.root = root
        self.root.title("Myelin Analyser")
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

        # Initialise 15x15 grid 
        self.grid_lines = []
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_resize) 
        self.detect_button = Button(root, text="Detect Pillar Centers", command=self.detect_pillar_centers)
        self.detect_button.grid(row=1, column=0, pady=5)



        # Add Dot and Delete Dot buttons

        self.add_dot_button = Button(root, text="Add Dot", command=self.enable_add_dot_mode)
        self.add_dot_button.grid(row=2, column=0, pady=5)

        self.delete_dot_button = Button(root, text="Delete Dot", command=self.enable_delete_dot_mode)
        self.delete_dot_button.grid(row=3, column=0, pady=5)

        self.count_button = Button(root, text="Count", command=self.save_cropped_images)
        self.count_button.grid(row=8, column=0, columnspan=2, pady=10)

        # Create Boxes button

        self.create_boxes_button = Button(root, text="Create Boxes", command=self.create_boxes)
        self.create_boxes_button.grid(row=4, column=0, pady=5)

        # Initialize add/delete dot mode flags

        self.add_dot_mode = False
        self.delete_dot_mode = False

        # Filter button

        # Sort button (initially disabled until filtering is applied)

        self.sort_button = Button(root, text="Sort", command=self.sort_images, state='disabled')
        self.sort_button.grid(row=6, column=0, pady=5)
        self.overlay_scale_x = self.myelin_image.shape[1] / self.display_myelin_image.shape[1]
        self.overlay_scale_y = self.myelin_image.shape[0] / self.display_myelin_image.shape[0]
        self.score_labels = {}
        self.score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_scored_label = Label(root, text="Total Scored: 0")
        self.total_scored_label.grid(row=9, column=0, columnspan=2)
        self.total_boxes_label = Label(root, text="Total Boxes Created: 0")
        self.total_boxes_label.grid(row=10, column=0, columnspan=2)

    def on_resize(self, event):
        # """Handle window resizing by updating the displayed cell image size."""
        # self.canvas.config(width=event.width, height=event.height)
        return



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

    def sort_images(self):
        """Extracts each boxed region from the overlay image in order for classification."""
        self.current_cell_index = 0
        self.grid_cells = []

        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            # Ensure we are extracting from the overlayed image
            cell_img = self.myelin_image[y1:y2, x1:x2]
            cell_img_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
            self.grid_cells.append(cell_img_pil)

        # Start classifying the first cell
        self.classify_cell()

    def detect_pillar_centers(self):

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

        else:

            messagebox.showinfo("Detection Complete", f"Detected {len(self.dot_positions)} pillars.")

    def enable_add_dot_mode(self):
        self.add_dot_mode = True
        self.delete_dot_mode = False

    def enable_delete_dot_mode(self):
        self.delete_dot_mode = True
        self.add_dot_mode = False

    def on_canvas_click(self, event):

        if self.add_dot_mode:
            # Add a dot at the click location and update both positions and visuals
            self.dot_positions.append((event.x, event.y))
            dot_id = self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="red")
            self.dot_visuals.append(dot_id)

        elif self.delete_dot_mode:
            if not self.dot_positions:
                return

            # Find and delete the closest dot both from the canvas and dot list

            if self.dot_positions:
                closest_dot_index = None
                min_dist = float("inf")

                # Find the dot with the minimum distance to the click position

                for i, (x, y) in enumerate(self.dot_positions):
                    dist = (x - event.x) ** 2 + (y - event.y) ** 2
                    if dist < min_dist:
                        closest_dot_index, min_dist = i, dist

                # If a dot is found close enough to the click, delete it

                if closest_dot_index is not None:
                    # Remove the dot from the canvas and update both lists
                    self.canvas.delete(self.dot_visuals[closest_dot_index])
                    del self.dot_positions[closest_dot_index]
                    del self.dot_visuals[closest_dot_index]


    def create_boxes(self):
        """Create boxes around detected centers and display scoring buttons."""

        box_size = 30
        self.canvas.delete("box")
        self.myelin_canvas.delete("box")
        self.box_positions = []  # Reset box positions
        self.scored_boxes = {}   # Dictionary to store scores for each box

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



        # Enable the scoring buttons

        self.total_boxes_label.config(text=f"Total Boxes Created: {len(self.box_positions)}")
        self.show_scoring_buttons()

    def show_scoring_buttons(self):

        """Display scoring buttons for the user to classify boxes."""

        self.score_buttons = []  # Store button references for later updates
        button_frame = Canvas(self.root)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        scores = {0: "red", 1: "orange", 2: "yellow", 3: "green"}

        for score, color in scores.items():

            button = Button(button_frame, text=f"Score {score}", bg=color,
                            command=lambda s=score: self.set_current_score(s))
            button.pack(side="left", padx=5)

            self.score_buttons.append(button)
            label = Label(button_frame, text=f"Count: {self.score_counts[score]}")
            label.pack(side="left", padx=5)
            self.score_labels[score] = label

        # Add an eraser button

        eraser_button = Button(button_frame, text="Eraser", bg="white", command=self.set_eraser_mode)
        eraser_button.pack(side="left", padx=5)
        self.eraser_button = eraser_button

    def set_eraser_mode(self):
        """Enable eraser mode to remove scores and highlights from boxes."""
        self.current_score = None  # No score is being set
        self.canvas.bind("<Button-1>", self.erase_box)

    def erase_box(self, event):
        """Erase the highlight and score of a selected box, replacing it with a blue fill."""
        x, y = event.x, event.y

        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Remove the score from the scored_boxes dictionary
                if i in self.scored_boxes:
                    del self.scored_boxes[i]

                # Clear any previous highlight for this box
                self.myelin_canvas.delete(f"box_{i}")

                # Redraw the box with a blue fill
                self.myelin_canvas.create_rectangle(
                    x1, y1, x2, y2, outline="blue", fill="", width=2, tags=f"box_{i}"
                )
                print(f"Box {i} erased and reset to blue outline.")
                break

    def set_current_score(self, score):
        """Set the current score and enable box selection."""
        self.current_score = score
        self.myelin_canvas.bind("<Button-1>", self.score_box)  # Bind clicks on the myelin canvas

    def score_box(self, event):
        """Score a box based on the current score or erase it if in eraser mode."""
        x, y = event.x, event.y
        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if self.current_score is None:
                    # Eraser mode: Remove score and reset highlight
                    if i in self.scored_boxes:
                        del self.scored_boxes[i]  # Remove from scored_boxes if present
                    self.myelin_canvas.delete(f"box_{i}")  # Clear previous highlight
                    self.myelin_canvas.create_rectangle(
                        x1, y1, x2, y2, outline="blue", fill="", width=2, tags=f"box_{i}"
                    )
                    print(f"Box {i} erased and reset to blue outline.")

                else:
                    # Assign the score and highlight the box
                    self.scored_boxes[i] = self.current_score
                    colors = {0: "red", 1: "orange", 2: "yellow", 3: "green"}
                    color = colors[self.current_score]
                    self.myelin_canvas.delete(f"box_{i}")  # Clear previous highlight
                    self.myelin_canvas.create_rectangle(
                        x1, y1, x2, y2, fill=color, stipple="gray25", outline="blue", width=2, tags=f"box_{i}"
                    )
                    print(f"Box {i} scored as {self.current_score} with {color} highlight.")

                break

        

        for score, count in self.score_counts.items():
            self.score_labels[score].config(text=f"Count: {count}")

        self.total_scored_label.config(text=f"Total Scored: {len(self.scored_boxes)}")

    def add_count_button(self):
        count_button = Button(self.root, text="Count", command=self.save_cropped_images)
        count_button.grid(row=8, column=0, columnspan=2, pady=10)

        

    def save_cropped_images(self):
        """Save cropped images to corresponding folders based on their scores."""
        if not self.scored_boxes:
            messagebox.showwarning("No Boxes Scored", "Please score at least one box before saving.")
            return

        # Load the overlay image for cropping

        base_image = cv2.imread(self.myelin_image_path, cv2.IMREAD_COLOR)

        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            if i in self.scored_boxes:  # Save only if the box has a score
                score = self.scored_boxes[i]
                # Scale coordinates back to the original overlay image size
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
                file_path = f"{score}/cell_{i}.png"
                cv2.imwrite(file_path, cropped_img)
                print(f"Saved {file_path}")

        messagebox.showinfo("Save Complete", "Scored boxes have been saved.")

    def get_cell_bounds(self, row, col):

        """Get the bounding box for a cell based on current grid line positions."""

        x1 = int(self.vertical_lines[col])
        y1 = int(self.horizontal_lines[row])
        x2 = int(self.vertical_lines[col + 1])
        y2 = int(self.horizontal_lines[row + 1])

        return x1, y1, x2, y2

    def update_box_highlight(self, index, color="white"):

        """Highlight the box at the given index with the specified color."""

        # Use box coordinates directly
        x1, y1, x2, y2 = self.box_positions[index]

        # Delete any previous highlights
        self.myelin_canvas.delete("highlight")

        # Draw a rectangle to highlight the box
        self.myelin_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3, tags="highlight")


if __name__ == "__main__":
    import sys
    
    # Check if running with command line arguments
    if len(sys.argv) == 3:
        # Simple approach: first arg is pillar, second is myelin
        pillar_image_path = sys.argv[1]
        myelin_image_path = sys.argv[2]
        
        print(f"Auto-loading images from command line:")
        print(f"  Pillar: {pillar_image_path}")
        print(f"  Myelin: {myelin_image_path}")
        
        # Verify files exist
        if not os.path.exists(pillar_image_path):
            print(f"ERROR: Pillar image not found: {pillar_image_path}")
            sys.exit(1)
        if not os.path.exists(myelin_image_path):
            print(f"ERROR: Myelin image not found: {myelin_image_path}")
            sys.exit(1)
        
        root = Tk()
        app = MyelinAnalyser(root, pillar_image_path, myelin_image_path)
        root.mainloop()
    else:
        # Original behavior - ask user to select files
        print(f"Manual mode: {len(sys.argv)} arguments provided")
        print(f"Usage: python ManualScore.py <pillar_path> <myelin_path>")
        
        root = Tk()
        root.withdraw()
        pillar_image_path = filedialog.askopenfilename(title="Select Pillar Image")
        myelin_image_path = filedialog.askopenfilename(title="Select Myelin Image")
        root.destroy()
        
        if pillar_image_path and myelin_image_path:
            root = Tk()
            app = MyelinAnalyser(root, pillar_image_path, myelin_image_path)
            root.mainloop()
        else:
            messagebox.showerror("Error", "Please select both pillar and myelin images.")