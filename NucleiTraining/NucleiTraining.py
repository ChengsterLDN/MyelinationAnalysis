from skimage.io import imread, imsave
from skimage.measure import label
import numpy as np
import glob
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEGMENTED_DIR = '.'          # folder containing your unlabelled segmented .tifs
MASKS_DIR     = 'masks'      # output folder for labelled masks
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MASKS_DIR, exist_ok=True)

tif_files = glob.glob(os.path.join(SEGMENTED_DIR, '*.tif'))

if not tif_files:
    print("No .tif files found in the current directory. Check your path.")
else:
    print(f"Found {len(tif_files)} segmented .tif file(s) to process.\n")

for tif_path in tif_files:
    filename = os.path.splitext(os.path.basename(tif_path))[0]  # e.g. "R1BenzDose6-1"
    output_path = os.path.join(MASKS_DIR, filename + '.png')

    # Check a matching raw image exists in images/
    raw_path = os.path.join('images', filename + '.png')
    if not os.path.exists(raw_path):
        print(f"  ⚠  Skipping '{filename}' — no matching raw image found at '{raw_path}'")
        continue

    # Load segmented mask
    binary_mask = imread(tif_path)

    # Normalise to binary (handles 0/255 from Fiji as well as 0/1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Label each nucleus with a unique integer (8-connectivity, safer for nuclei)
    instance_labels = label(binary_mask, connectivity=2).astype(np.uint16)
    n_nuclei = instance_labels.max()

    # Save labelled mask as .png (uint16, matching raw image filename)
    imsave(output_path, instance_labels)
    print(f"  ✓  {filename}.tif  →  {output_path}  ({n_nuclei} nuclei labelled)")

print("\nDone! Your StarDist training folders are ready:")
print("  images/   ← raw fluorescence .png files")
print("  masks/    ← integer instance label .png files (uint16)")