import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tkinter import Tk, filedialog, messagebox
from skimage import io
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def setup_gpu():
    # ensure GPU hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected — running on CPU")


class NucleiStarDist:
    def __init__(self, image_path, output_folder, model_type='2D_versatile_fluo'):
        self.image_path = image_path
        self.output_folder = output_folder
        self.model_type = model_type
        self.image = None
        self.nuclei_count = []   # List of coordinate arrays (one per nucleus)
        self.nuclei_prop = []    # List of property dicts, matching NucleiAnalysis format
        self.labels = None

    def load_image(self):
        """Load the image from disk."""
        print(f"Loading image from: {self.image_path}")
        self.image = io.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

    def detect_nuclei(self, model):
        """
        Normalise the image and run StarDist prediction.
        Extracts centroid, area and circularity for each detected nucleus,
        mirroring the property dict structure used in NucleiAnalysis.py.
        """
        img = self.image

        # Use only the first channel if multi-channel (e.g. RGB fluorescence)
        if img.ndim == 3 and self.model_type == '2D_versatile_fluo':
            print("Multi-channel image detected — using first channel.")
            img = img[:, :, 0]

        # Normalise between 1st and 99.8th percentile (StarDist standard)
        print("Normalising image...")
        img_normalised = normalize(img, 1, 99.8, axis=(0, 1))

        # Run StarDist prediction
        print("Running StarDist prediction...")
        self.labels, details = model.predict_instances(img_normalised)

        coords = details['coord']   # shape: (N, 2, n_rays) — polygon vertices per nucleus
        self.nuclei_count = []
        self.nuclei_prop = []

        for i, coord in enumerate(coords):
            # coord shape is (2, n_rays): row (y) and col (x) arrays
            ys, xs = coord[0], coord[1]

            # Centroid from mean of polygon vertices
            xc = int(np.mean(xs))
            yc = int(np.mean(ys))

            # Approximate area from polygon using the shoelace formula
            n = len(xs)
            area = 0.5 * abs(
                sum(xs[j] * ys[(j + 1) % n] - xs[(j + 1) % n] * ys[j] for j in range(n))
            )

            # Approximate perimeter by summing edge lengths between polygon vertices
            perimeter = sum(
                np.sqrt((xs[(j + 1) % n] - xs[j]) ** 2 + (ys[(j + 1) % n] - ys[j]) ** 2)
                for j in range(n)
            )

            # Circularity (1.0 = perfect circle)
            if perimeter > 0 and area > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0.0

            self.nuclei_count.append(coord)

            properties = {
                "nuclei_id": i,
                "x_c": xc,
                "y_c": yc,
                "area": float(area),
                "circularity": float(circularity)
            }
            self.nuclei_prop.append(properties)

        # Derive output name from the parent folder
        if os.path.isdir(self.image_path):
            parent_folder_name = os.path.basename(os.path.normpath(self.image_path))
        else:
            parent_folder_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # Save properties to JSON
        json_path = os.path.join(self.output_folder, f"{parent_folder_name}_nuclei_props.json")
        with open(json_path, 'w') as json_file:
            json.dump(self.nuclei_prop, json_file, indent=4)

        print(f"Detected {len(self.nuclei_count)} nuclei")
        print(f"Nuclei properties saved to {json_path}")

        return len(self.nuclei_count) > 0

    def visualise(self):
        #Overlay StarDist segmentation on the original image and save a PNG
        if self.labels is None or not self.nuclei_prop:
            return

        # Use first channel for display if multi-channel
        display_img = self.image[:, :, 0] if self.image.ndim == 3 else self.image

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].set_title("Original Image")
        axes[0].imshow(display_img, cmap='gray')
        axes[0].axis('off')

        axes[1].set_title(f"StarDist Segmentation\nDetected Nuclei: {len(self.nuclei_count)}")
        axes[1].imshow(display_img, cmap='gray')
        masked_labels = np.ma.masked_where(self.labels == 0, self.labels)
        axes[1].imshow(masked_labels, cmap='nipy_spectral', alpha=0.5)

        # Draw centroids and ID labels, matching NucleiAnalysis.py style
        for props in self.nuclei_prop:
            axes[1].plot(props["x_c"], props["y_c"], 'bo', markersize=3)
            axes[1].text(props["x_c"] + 6, props["y_c"], str(props["nuclei_id"]),
                         color='red', fontsize=5)
        axes[1].axis('off')

        plt.tight_layout()

        # Derive output name from image path
        parent_folder_name = os.path.splitext(os.path.basename(self.image_path))[0]
        vis_path = os.path.join(self.output_folder, f"{parent_folder_name}_segmentation.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Segmentation image saved to {vis_path}")

    def process(self, model):
        """Full pipeline: load → detect → visualise."""
        self.load_image()

        if not self.detect_nuclei(model):
            print("No nuclei detected in image.")
            return False

        self.visualise()
        return True


def process_all_subfolders(parent_directory, model_type='2D_versatile_fluo'):
    """
    Walk every subfolder in parent_directory, find nuclei_mip.png,
    and run StarDist analysis — mirroring NucleiAnalysis.py's folder logic.
    The model is loaded once and reused for all images.
    """
    print(f"Loading pretrained StarDist model: {model_type}...")
    model = StarDist2D.from_pretrained(model_type)
    model = StarDist2D(None, name='SDFTv1', basedir='models')

    processed_folders = 0
    successful_folders = 0

    for folder_name in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, folder_name)

        if os.path.isdir(subfolder_path):
            nuclei_image_path = os.path.join(subfolder_path, "nuclei_mip.png")

            if os.path.exists(nuclei_image_path):
                print(f"\nProcessing folder: {folder_name}")
                processed_folders += 1

                try:
                    analyser = NucleiStarDist(nuclei_image_path, subfolder_path, model_type)
                    success = analyser.process(model)

                    if success:
                        successful_folders += 1
                        print(f"✓ Successfully processed {folder_name} "
                              f"— Found {len(analyser.nuclei_count)} nuclei")
                    else:
                        print(f"✗ No nuclei detected in {folder_name}")

                except Exception as e:
                    print(f"✗ Error processing {folder_name}: {str(e)}")
            else:
                print(f"Skipping {folder_name}: nuclei_mip.png not found")

    return processed_folders, successful_folders


if __name__ == "__main__":
    setup_gpu()

    root = Tk()
    root.withdraw()

    parent_directory = filedialog.askdirectory(
        title="Select Parent Directory Containing Subfolders"
    )

    if not parent_directory:
        messagebox.showerror("Error", "Please select a parent directory.")
        exit()

    # Choose '2D_versatile_fluo' for fluorescence or '2D_versatile_he' for H&E
    MODEL_TO_USE = '2D_versatile_fluo'

    try:
        processed, successful = process_all_subfolders(parent_directory, MODEL_TO_USE)

        messagebox.showinfo(
            "Processing Complete",
            f"Processed {processed} folders\n"
            f"Successfully completed: {successful}\n"
            f"Failed: {processed - successful}"
        )

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

    root.destroy()