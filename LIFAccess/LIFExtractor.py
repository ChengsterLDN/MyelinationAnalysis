import sys
import os
import glob
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from readlif.reader import LifFile
import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors
import re

class LIFProcessor:
    """Class to process LIF files and extract images as PNGs"""
    
    def __init__(self):
        self.channel_names = {0: "nuclei", 1: "mbp", 2: "pillar"}
        self.yellow_cmap = mcolors.LinearSegmentedColormap.from_list('yellow_cmap', ['black', 'yellow'])
        self.cyan_cmap = mcolors.LinearSegmentedColormap.from_list('cyan_cmap', ['black', 'cyan'])
    
    def clean_filename(self, filename):
        """Clean filename by removing or replacing invalid characters"""
        # Remove trailing spaces
        filename = filename.strip()
        # Replace invalid characters with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        return filename
    
    def browse_lif_file(self):
        """Open file dialog to browse for LIF files"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_paths = filedialog.askopenfilenames(
            title="Select LIF file(s)",
            filetypes=[("LIF files", "*.lif"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_paths
    
    def process_lif_file(self, lif_file_path):
        """Process a single LIF file"""
        try:
            print(f"Processing {os.path.basename(lif_file_path)}")
            lif_file = LifFile(lif_file_path)

            base_name = os.path.splitext(os.path.basename(lif_file_path))[0]
            base_name = self.clean_filename(base_name)
            output_dir = f'./{base_name}_extracted_pngs'
            os.makedirs(output_dir, exist_ok=True)

            all_images = list(lif_file.get_iter_image())
            print(f"Found {len(all_images)} images in LIF file")

            for img_index, lif_image in enumerate(all_images):
                series_name = self.clean_filename(lif_image.name)
                print(f"Processing image {img_index + 1}/{len(all_images)}: {series_name}")
                dims = lif_image.dims
                print(f"  Dimensions: {dims}")
                print(f"  Channels: {lif_image.channels}")
                
                # Create series folder first, then channel subfolders within it
                series_dir = os.path.join(output_dir, series_name)
                os.makedirs(series_dir, exist_ok=True)

                # Create channel subfolders within the series folder
                for channel in [0, 1, 2]:
                    if channel < lif_image.channels:
                        channel_dir = os.path.join(output_dir, series_name, self.channel_names[channel])
                        os.makedirs(channel_dir, exist_ok=True)

                for channel in [0, 1, 2]:
                    if channel >= lif_image.channels:
                        print(f"  Channel {channel} not available (only {lif_image.channels} channels)")
                        continue

                    for z in range(dims.z if hasattr(dims, 'z') else 1):
                        for t in range(dims.t if hasattr(dims, 't') else 1):
                            try:
                                pil_image = lif_image.get_frame(t=t, z=z, c=channel)
                                np_image = np.array(pil_image)

                                if np_image.max() > np_image.min():
                                    normalized_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
                                else:
                                    normalized_image = np_image
                                
                                if channel == 2:
                                    colored_image = self.yellow_cmap(normalized_image)
                                    channel_name = "yellow"
                                else:  # channels 0 and 1
                                    colored_image = self.cyan_cmap(normalized_image)
                                    channel_name = "cyan"
                                
                                rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
                                img = Image.fromarray(rgb_image, 'RGB')
                                
                                # Determine channel directory with series subfolder and filename
                                channel_dir = os.path.join(output_dir, series_name, self.channel_names[channel])
                                filename = f'{series_name}_c{channel}_{channel_name}_z{z}_t{t}.png'
                                filename = self.clean_filename(filename)
                                filepath = os.path.join(channel_dir, filename)

                                img.save(filepath, format='PNG')
                                print(f"    Saved: {os.path.join(self.channel_names[channel], series_name, filename)}")
                                
                            except Exception as e:
                                print(f"    Error processing frame: {e}")
            
            print(f"Finished processing {os.path.basename(lif_file_path)}")
            return True
            
        except Exception as e:
            print(f"Error processing {lif_file_path}: {e}")
            return False

    def main(self):
        """Main function - handles file selection and processing"""
        print("LIF to PNG Extractor")
        print("=" * 50)

        # Browse for LIF files
        lif_files = self.browse_lif_file()
        
        if not lif_files:
            print("No files selected!")
            input("Press Enter to exit...")
            return
        
        print(f"Selected {len(lif_files)} LIF file(s):")
        for lif_file in lif_files:
            print(f"  - {os.path.basename(lif_file)}")
        
        print("\nStarting processing...")
        for lif_file in lif_files:
            if lif_file.lower().endswith('.lif'):
                self.process_lif_file(lif_file)
            else:
                print(f"Skipping non-LIF file: {os.path.basename(lif_file)}")
        
        print("\nProcessing complete!")
        input("Press Enter to exit...")

if __name__ == "__main__":
    processor = LIFProcessor()
    processor.main()