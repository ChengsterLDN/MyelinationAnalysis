import cv2
import os
import numpy as np
from tkinter import Tk, Button, Canvas, Label, messagebox, filedialog
from PIL import Image, ImageTk
from scipy.ndimage import convolve
import sys
import json
import tempfile

class MyelinAnalyser:

    def __init__(self, root, pillar_image_path, myelin_image_path):
        self.root = root
        self.root.title("Myelin Analyser")
        
        self.pillar_image_path = pillar_image_path        
        self.myelin_image_path = myelin_image_path
        
        # Identifier for saving results (Use the folder name)
        self.sample_name = os.path.basename(os.path.dirname(pillar_image_path))
        
        self.dot_positions = []
        self.box_positions = []  
        self.scored_boxes = {} # Dictionary to store scores {box_index: score}
        self.current_cell_index = 0  
        
        self.pillar_image = cv2.imread(pillar_image_path, cv2.IMREAD_COLOR)
        self.myelin_image = cv2.imread(myelin_image_path, cv2.IMREAD_COLOR)
        
        # Resize images for display
        self.display_pillar_image = self.resize_image(self.pillar_image)
        self.display_myelin_image = self.resize_image(self.myelin_image)
        self.display_pillar_scale = self.pillar_image.shape[1] / self.display_pillar_image.shape[1]

        # UI Setup
        self.canvas = Canvas(root, width=self.display_pillar_image.shape[1], height=self.display_pillar_image.shape[0])
        self.canvas.grid(row=0, column=0)
        self.pillar_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_pillar_image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.pillar_photo)

        self.myelin_canvas = Canvas(root, width=self.display_myelin_image.shape[1], height=self.display_myelin_image.shape[0])
        self.myelin_canvas.grid(row=0, column=1)
        self.myelin_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_myelin_image, cv2.COLOR_BGR2RGB)))
        self.myelin_canvas.create_image(0, 0, anchor='nw', image=self.myelin_photo)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Buttons
        self.detect_button = Button(root, text="Detect Pillar Centers", command=self.detect_pillar_centers)
        self.detect_button.grid(row=1, column=0, pady=5)

        self.add_dot_button = Button(root, text="Add Dot", command=self.enable_add_dot_mode)
        self.add_dot_button.grid(row=2, column=0, pady=5)

        self.delete_dot_button = Button(root, text="Delete Dot", command=self.enable_delete_dot_mode)
        self.delete_dot_button.grid(row=3, column=0, pady=5)

        self.create_boxes_button = Button(root, text="Create Boxes", command=self.create_boxes)
        self.create_boxes_button.grid(row=4, column=0, pady=5)

        # Save Button (Modified to save to temp file and close)
        self.save_button = Button(root, text="Save & Close", command=self.save_and_close, bg="green", fg="white")
        self.save_button.grid(row=8, column=0, columnspan=2, pady=10)

        self.add_dot_mode = False
        self.delete_dot_mode = False
        self.current_score = None
        
        self.overlay_scale_x = self.myelin_image.shape[1] / self.display_myelin_image.shape[1]
        self.overlay_scale_y = self.myelin_image.shape[0] / self.display_myelin_image.shape[0]
        
        self.score_labels = {}
        self.score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        self.total_scored_label = Label(root, text="Total Scored: 0")
        self.total_scored_label.grid(row=9, column=0, columnspan=2)
        self.total_boxes_label = Label(root, text="Total Boxes Created: 0")
        self.total_boxes_label.grid(row=10, column=0, columnspan=2)
        
        # Get temp file path from command line or create one
        if len(sys.argv) > 3:
            self.temp_output_file = sys.argv[3]
        else:
            # Create a temporary file name based on sample name
            temp_dir = tempfile.gettempdir()
            self.temp_output_file = os.path.join(temp_dir, f"manualscore_{os.path.basename(self.sample_name)}.json")
        
        # Delete any existing temp file
        if os.path.exists(self.temp_output_file):
            os.remove(self.temp_output_file)

    def resize_image(self, image, max_size=600):
        h, w = image.shape[:2]
        if h <= max_size and w <= max_size:
            return image
        scale = min(max_size / h, max_size / w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def detect_pillar_centers(self):
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
            if area < min_area: continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 or x+w >= img_w or y+h >= img_h: continue

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                display_x, display_y = int(center_x / self.display_pillar_scale), int(center_y / self.display_pillar_scale)
                self.dot_positions.append((display_x, display_y))
                self.dot_visuals.append(self.canvas.create_oval(display_x - 2, display_y - 2, display_x + 2, display_y + 2, fill="red"))
        
        messagebox.showinfo("Detection Complete", f"Detected {len(self.dot_positions)} pillars.")

    def enable_add_dot_mode(self):
        self.add_dot_mode = True
        self.delete_dot_mode = False

    def enable_delete_dot_mode(self):
        self.delete_dot_mode = True
        self.add_dot_mode = False

    def on_canvas_click(self, event):
        if self.add_dot_mode:
            self.dot_positions.append((event.x, event.y))
            dot_id = self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="red")
            self.dot_visuals.append(dot_id)

        elif self.delete_dot_mode:
            if not self.dot_positions: return
            closest_dot_index = None
            min_dist = float("inf")
            for i, (x, y) in enumerate(self.dot_positions):
                dist = (x - event.x) ** 2 + (y - event.y) ** 2
                if dist < min_dist:
                    closest_dot_index, min_dist = i, dist

            if closest_dot_index is not None:
                self.canvas.delete(self.dot_visuals[closest_dot_index])
                del self.dot_positions[closest_dot_index]
                del self.dot_visuals[closest_dot_index]

    def create_boxes(self):
        box_size = 30
        self.canvas.delete("box")
        self.myelin_canvas.delete("box")
        self.box_positions = []
        self.scored_boxes = {}

        self.dot_positions = sorted(self.dot_positions, key=lambda pos: (pos[1], pos[0]))

        for (x, y) in self.dot_positions:
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(self.display_myelin_image.shape[1], x + box_size // 2)
            y2 = min(self.display_myelin_image.shape[0], y + box_size // 2)
            self.box_positions.append((x1, y1, x2, y2))
            self.myelin_canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, tags="box")

        self.total_boxes_label.config(text=f"Total Boxes Created: {len(self.box_positions)}")
        self.show_scoring_buttons()

    def show_scoring_buttons(self):
        self.score_buttons = []
        button_frame = Canvas(self.root)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        scores = {0: "red", 1: "orange", 2: "yellow", 3: "green"}

        for score, color in scores.items():
            button = Button(button_frame, text=f"Score {score}", bg=color,
                            command=lambda s=score: self.set_current_score(s))
            button.pack(side="left", padx=5)
            self.score_buttons.append(button)
            
            # Label
            label = Label(button_frame, text=f"Count: 0")
            label.pack(side="left", padx=5)
            self.score_labels[score] = label

        eraser_button = Button(button_frame, text="Eraser", bg="white", command=self.set_eraser_mode)
        eraser_button.pack(side="left", padx=5)

    def set_eraser_mode(self):
        self.current_score = None
        self.myelin_canvas.bind("<Button-1>", self.erase_box)

    def erase_box(self, event):
        x, y = event.x, event.y
        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if i in self.scored_boxes:
                    del self.scored_boxes[i]
                self.myelin_canvas.delete(f"box_{i}")
                self.myelin_canvas.create_rectangle(x1, y1, x2, y2, outline="blue", fill="", width=2, tags=f"box_{i}")
                self.update_score_counts()
                break

    def set_current_score(self, score):
        self.current_score = score
        self.myelin_canvas.bind("<Button-1>", self.score_box)

    def score_box(self, event):
        x, y = event.x, event.y
        for i, (x1, y1, x2, y2) in enumerate(self.box_positions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.scored_boxes[i] = self.current_score
                colors = {0: "red", 1: "orange", 2: "yellow", 3: "green"}
                color = colors[self.current_score]
                self.myelin_canvas.delete(f"box_{i}")
                self.myelin_canvas.create_rectangle(x1, y1, x2, y2, fill=color, stipple="gray25", outline="blue", width=2, tags=f"box_{i}")
                break
        self.update_score_counts()

    def update_score_counts(self):
        # Recalculate totals
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in self.scored_boxes.values():
            counts[s] = counts.get(s, 0) + 1
        
        # Update UI
        for score, count in counts.items():
            if score in self.score_labels:
                self.score_labels[score].config(text=f"Count: {count}")
        
        self.total_scored_label.config(text=f"Total Scored: {len(self.scored_boxes)}")

    def save_and_close(self):
        """Save scores to a temporary JSON file and close the window."""
        if not self.scored_boxes:
            response = messagebox.askyesno("No Boxes Scored", 
                                          "No boxes have been scored. Are you sure you want to close without saving?")
            if not response:
                return
            else:
                # Create empty scores if user wants to close without scoring
                counts = {0: 0, 1: 0, 2: 0, 3: 0}
        else:
            # Calculate final counts
            counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for s in self.scored_boxes.values():
                counts[s] = counts.get(s, 0) + 1
        
        # Prepare data to save
        scores_data = {
            'sample_name': self.sample_name,
            'pillar_image': self.pillar_image_path,
            'myelin_image': self.myelin_image_path,
            'scores': {
                '0': counts[0],
                '1': counts[1],
                '2': counts[2],
                '3': counts[3]
            },
            'total_scored': len(self.scored_boxes),
            'total_boxes': len(self.box_positions),
            'status': 'completed'
        }
        
        # Save to temporary file
        try:
            with open(self.temp_output_file, 'w') as f:
                json.dump(scores_data, f, indent=2)
            print(f"Scores saved to temporary file: {self.temp_output_file}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save scores: {e}")
        
        # Close the window
        self.root.destroy()

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pillar_image_path = sys.argv[1]
        myelin_image_path = sys.argv[2]
        
        if not os.path.exists(pillar_image_path) or not os.path.exists(myelin_image_path):
            print("Error: Images not found.")
            sys.exit(1)
            
        root = Tk()
        
        # Pass temp file path if provided
        if len(sys.argv) == 4:
            app = MyelinAnalyser(root, pillar_image_path, myelin_image_path)
        else:
            app = MyelinAnalyser(root, pillar_image_path, myelin_image_path)
            
        root.mainloop()
    else:
        root = Tk()
        root.withdraw()
        pillar = filedialog.askopenfilename(title="Select Pillar Image")
        myelin = filedialog.askopenfilename(title="Select Myelin Image")
        root.destroy()
        
        if pillar and myelin:
            root = Tk()
            app = MyelinAnalyser(root, pillar, myelin)
            root.mainloop()