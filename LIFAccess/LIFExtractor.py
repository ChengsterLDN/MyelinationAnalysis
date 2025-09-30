import sys
import os
import glob
from PIL import Image
from readlif.reader import LifFile
import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors

def process_lif_file(lif_file_path):
    """Process a single LIF file"""
    try:
        print(f"Processing {os.path.basename(lif_file_path)}")
        lif_file = LifFile(lif_file_path)

        base_name = os.path.splitext(os.path.basename(lif_file_path))[0]
        output_dir = f'./{base_name}_extracted_pngs'
        os.makedirs(output_dir, exist_ok=True)

        yellow_cmap = mcolors.LinearSegmentedColormap.from_list('yellow_cmap', ['black', 'yellow'])
        cyan_cmap = mcolors.LinearSegmentedColormap.from_list('cyan_cmap', ['black', 'cyan'])

        all_images = list(lif_file.get_iter_image())
        print(f"Found {len(all_images)} images in LIF file")

        for img_index, lif_image in enumerate(all_images):
            print(f"Processing image {img_index + 1}/{len(all_images)}")
            dims = lif_image.dims
            print(f"  Dimensions: {dims}")
            print(f"  Channels: {lif_image.channels}")
            # Create main directory for this image
            img_dir = os.path.join(output_dir, f'image_{img_index}')
            os.makedirs(img_dir, exist_ok=True)
            
            # Create subdirectories for each channel
            channel_names = {0: "nuclei", 1: "mbp", 2: "pillar"}
            for channel in [0, 1, 2]:
                if channel >= lif_image.channels:
                    continue 
                channel_dir = os.path.join(img_dir, channel_names[channel])
                os.makedirs(channel_dir, exist_ok=True)

            for channel in [0, 1, 2]:
                if channel > lif_image.channels:
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
                                colored_image = yellow_cmap(normalized_image)
                                channel_name = "yellow"
                            elif channel == 1:
                                colored_image = cyan_cmap(normalized_image)
                                channel_name = "cyan"

                            elif channel == 0:
                                colored_image = cyan_cmap(normalized_image)
                                channel_name = "cyan"
                            
                            rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_image, 'RGB')
                            
                            # Determine channel directory and filename
                            channel_dir = os.path.join(img_dir, channel_names[channel])
                            filename = f'img_{img_index}_c{channel}_{channel_name}_z{z}_t{t}.png'
                            filepath = os.path.join(channel_dir, filename)

                            img.save(filepath, format='PNG')
                            print(f"    Saved: {filename}")
                            
                        except Exception as e:
                            print(f"    Error: {e}")
        
        print(f"Finished processing {os.path.basename(lif_file_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing {lif_file_path}: {e}")
        return False

def main():
    """Main function - handles drag & drop and directory processing"""
    print("LIF to PNG Extractor")
    print("=" * 50)

    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            if file_path.lower().endswith('.lif'):
                process_lif_file(file_path)
            else:
                print(f"Skipping non-LIF file: {os.path.basename(file_path)}")
    else:

        lif_files = glob.glob('*.lif')
        if not lif_files:
            print("No LIF files found in current directory!")
            input("Press Enter to exit...")
            return
        
        print(f"Found {len(lif_files)} LIF file(s) in current directory:")
        for lif_file in lif_files:
            print(f"  - {lif_file}")
        
        print("\nStarting processing...")
        for lif_file in lif_files:
            process_lif_file(lif_file)
    
    print("\nProcessing complete!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()