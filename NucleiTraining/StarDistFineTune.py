"""
Fine-tune the pretrained StarDist fluorescence model ('2D_versatile_fluo')
on your own nuclei data.

Folder structure expected (run from parent folder):
    images/   ← raw fluorescence .png files
    masks/    ← integer instance label .png files (uint16, from NucleiTraining.py)

Install dependencies:
    pip install stardist tensorflow scikit-image tifffile
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize

from stardist import fill_label_holes, calculate_extents
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

# ── CONFIG 
IMAGES_DIR      = 'images'
MASKS_DIR       = 'masks'
MODEL_SAVE_DIR  = 'models'        
MODEL_NAME      = 'stardist_finetuned'

PRETRAINED_MODEL = '2D_versatile_fluo'  #

VAL_FRACTION    = 0.15              # fraction of data held out for validation
TRAIN_EPOCHS    = 10               
TRAIN_STEPS     = 100              
BATCH_SIZE      = 4                 # reduce to 2 if out-of-memory error
USE_GPU         = False             # OpenCL not supported for WSL2
AUGMENT         = True              # flips/rotations during training

NORM_PERCENTILE_LOW  = 1            # percentile for image normalisation
NORM_PERCENTILE_HIGH = 99.8

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

image_paths = sorted(glob(os.path.join(IMAGES_DIR, '*.png')))
mask_paths  = sorted(glob(os.path.join(MASKS_DIR,  '*.png')))

assert len(image_paths) > 0, f"No images found in '{IMAGES_DIR}/'"
assert len(image_paths) == len(mask_paths), (
    f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks.\n"
    "Make sure every image has a matching mask with the same filename."
)

print(f"Found {len(image_paths)} image/mask pairs.")

def load_images(paths):
    imgs = []
    for p in tqdm(paths, desc="Loading"):
        img = imread(p)
        # Convert RGB/RGBA to grayscale if needed
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., 0]   # take first channel; adjust if your signal is elsewhere
        imgs.append(img.astype(np.float32))
    return imgs

def load_masks(paths):
    masks = []
    for p in tqdm(paths, desc="Loading"):
        mask = imread(p).astype(np.uint16)
        mask = fill_label_holes(mask)   # fill any holes left in labelled nuclei
        masks.append(mask)
    return masks

print("\nLoading images...")
X = load_images(image_paths)
print("Loading masks...")
Y = load_masks(mask_paths)

# NORMALISE IMAGES 
print("\nNormalising images...")
X = [normalize(x, NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH) for x in X]

# TRAIN / VALIDATION SPLIT
n_val = max(1, int(len(X) * VAL_FRACTION))
rng   = np.random.default_rng(42)
idx   = rng.permutation(len(X))

idx_val   = list(idx[:n_val])
idx_train = list(idx[n_val:])

X_val,   Y_val   = [X[i] for i in idx_val],   [Y[i] for i in idx_val]
X_train, Y_train = [X[i] for i in idx_train], [Y[i] for i in idx_train]

print(f"\nSplit → {len(X_train)} training, {len(X_val)} validation images")

# LOAD PRETRAINED MODEL AND CONFIGURE FOR FINE-TUNING
print(f"\nLoading pretrained model '{PRETRAINED_MODEL}'...")
model = StarDist2D.from_pretrained(PRETRAINED_MODEL)

# Update the model's save location and name
model.basedir = MODEL_SAVE_DIR
model.name    = MODEL_NAME

# Fine-tuning config — keeps the pretrained architecture, just adjusts training
model.config.train_epochs          = TRAIN_EPOCHS
model.config.train_steps_per_epoch = TRAIN_STEPS
model.config.train_batch_size      = BATCH_SIZE
model.config.use_gpu               = USE_GPU

# Augmentation: random rotations and flips
if AUGMENT:
    model.config.train_augment_axes = (0, 1)   # flip along X and Y axes

print("\nModel config:")
print(f"  Epochs:          {model.config.train_epochs}")
print(f"  Steps per epoch: {model.config.train_steps_per_epoch}")
print(f"  Batch size:      {model.config.train_batch_size}")
print(f"  Augmentation:    {AUGMENT}")

# SANITY CHECK: MEDIAN OBJECT SIZE 
extents = calculate_extents(Y_train)
anisotropy = tuple(np.max(extents) / extents)
print(f"\nMedian nucleus extent (px): {extents}  |  anisotropy: {anisotropy}")
if max(anisotropy) > 2.5:
    print("  High anisotropy detected — consider checking your data")

# TRAIN 
print("\nStarting fine-tuning...\n")
model.train(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    augmenter=None,          # StarDist handles augmentation internally via config
    seed=42,
)

# OPTIMISE THRESHOLDS ON VALIDATION SET
print("\nOptimising probability and NMS thresholds on validation set...")
model.optimize_thresholds(X_val, Y_val)

# SAVE 
model.export_TF(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + '_TF'))
print(f"\n✓ Fine-tuned model saved to:  {MODEL_SAVE_DIR}/{MODEL_NAME}")
print(f"✓ TensorFlow export saved to: {MODEL_SAVE_DIR}/{MODEL_NAME}_TF")


"""
To reload:

    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    model = StarDist2D(None, name='stardist_finetuned', basedir='models')
    img   = normalize(imread('my_image.png'), 1, 99.8)
    labels, details = model.predict_instances(img)
"""