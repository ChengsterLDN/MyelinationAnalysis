"""
Fine-tune the pretrained StarDist fluorescence model ('2D_versatile_fluo')

Folder structure expected (place images/ and masks/ alongside this script):
    images/   ← raw fluorescence .png files
    masks/    ← integer instance label .png files (uint16, from NucleiTraining.py)
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from skimage.io import imread
from stardist import fill_label_holes, calculate_extents
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ── PATHS — all relative to THIS script's location, not the working directory ──
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR      = os.path.join(SCRIPT_DIR, 'images')
MASKS_DIR       = os.path.join(SCRIPT_DIR, 'masks')
MODEL_SAVE_DIR  = os.path.join(SCRIPT_DIR, 'models')
MODEL_NAME      = 'SDFTv1'

print(f"Script directory:   {SCRIPT_DIR}")
print(f"Images directory:   {IMAGES_DIR}")
print(f"Masks directory:    {MASKS_DIR}")
print(f"Model will save to: {os.path.join(MODEL_SAVE_DIR, MODEL_NAME)}")

# ── CONFIG ──────────────────────────────────────────────────────────────────────
PRETRAINED_MODEL     = '2D_versatile_fluo'
VAL_FRACTION         = 0.15
TRAIN_EPOCHS         = 3
TRAIN_STEPS          = 100
BATCH_SIZE           = 4        # reduce to 2 if you get out-of-memory errors
AUGMENT              = True

NORM_PERCENTILE_LOW  = 1
NORM_PERCENTILE_HIGH = 99.8

# ── GPU SETUP ───────────────────────────────────────────────────────────────────
# # This doesn't works for WSL2 as StarDist uses OpenCL. Basically train on CPU :(
"""def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled — training will use GPU.")
        return True
    else:
        print("\nNo GPU detected — training will run on CPU.")
        print("  If you have a GPU, check your CUDA/cuDNN installation.")
        return False

USE_GPU = setup_gpu()

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'"""



# ── LOAD DATA ───────────────────────────────────────────────────────────────────
image_paths = sorted(glob(os.path.join(IMAGES_DIR, '*.png')))
mask_paths  = sorted(glob(os.path.join(MASKS_DIR,  '*.png')))

assert len(image_paths) > 0, (
    f"No images found in '{IMAGES_DIR}'\n"
    f"Make sure your images/ folder is in the same directory as this script."
)
assert len(image_paths) == len(mask_paths), (
    f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks.\n"
    "Every image must have a matching mask with the same filename."
)

print(f"\nFound {len(image_paths)} image/mask pairs.")


def load_images(paths):
    imgs = []
    for p in tqdm(paths, desc="Loading images"):
        img = imread(p)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., 0]   # take first channel; adjust if your signal is elsewhere
        imgs.append(img.astype(np.float32))
    return imgs


def load_masks(paths):
    masks = []
    for p in tqdm(paths, desc="Loading masks"):
        mask = imread(p).astype(np.uint16)
        mask = fill_label_holes(mask)
        masks.append(mask)
    return masks


print("\nLoading images...")
X = load_images(image_paths)
print("Loading masks...")
Y = load_masks(mask_paths)

# ── NORMALISE ───────────────────────────────────────────────────────────────────
print("\nNormalising images...")
X = [normalize(x, NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH) for x in X]

# ── TRAIN / VALIDATION SPLIT ────────────────────────────────────────────────────
n_val = max(1, int(len(X) * VAL_FRACTION))
rng   = np.random.default_rng(42)
idx   = rng.permutation(len(X))

idx_val   = list(idx[:n_val])
idx_train = list(idx[n_val:])

X_val,   Y_val   = [X[i] for i in idx_val],   [Y[i] for i in idx_val]
X_train, Y_train = [X[i] for i in idx_train], [Y[i] for i in idx_train]

print(f"\nSplit → {len(X_train)} training, {len(X_val)} validation images")

# ── LOAD PRETRAINED MODEL ────────────────────────────────────────────────────────
print(f"\nLoading pretrained model '{PRETRAINED_MODEL}'...")
pretrained = StarDist2D.from_pretrained(PRETRAINED_MODEL)

# Create model at the correct save location, copying pretrained config
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model = StarDist2D(pretrained.config, name=MODEL_NAME, basedir=MODEL_SAVE_DIR)

# Copy pretrained weights into the new model
model.keras_model.set_weights(pretrained.keras_model.get_weights())
print(f"Weights loaded from pretrained. Saving to: {model.logdir}")

# Fine-tuning config
model.config.train_epochs          = TRAIN_EPOCHS
model.config.train_steps_per_epoch = TRAIN_STEPS
model.config.train_batch_size      = BATCH_SIZE
#model.config.use_gpu               = USE_GPU

if AUGMENT:
    model.config.train_augment_axes = (0, 1)

print("\nModel config:")
print(f"  Pretrained base:  {PRETRAINED_MODEL}")
print(f"  Epochs:           {model.config.train_epochs}")
print(f"  Steps per epoch:  {model.config.train_steps_per_epoch}")
print(f"  Batch size:       {model.config.train_batch_size}")
#print(f"  GPU:              {USE_GPU}")
print(f"  Augmentation:     {AUGMENT}")

# ── SANITY CHECK ─────────────────────────────────────────────────────────────────
extents    = calculate_extents(Y_train)
anisotropy = tuple(np.max(extents) / extents)
print(f"\nMedian nucleus extent (px): {extents}  |  anisotropy: {anisotropy}")
if max(anisotropy) > 2.5:
    print("  ⚠ High anisotropy detected — consider checking your data")

# ── TRAIN ────────────────────────────────────────────────────────────────────────
print("\nStarting fine-tuning...\n")
model.train(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    augmenter=None,   # StarDist handles augmentation internally via config
    seed=42,
)

# ── OPTIMISE THRESHOLDS ──────────────────────────────────────────────────────────
print("\nOptimising probability and NMS thresholds on validation set...")
model.optimize_thresholds(X_val, Y_val)

# ── CONFIRM SAVE LOCATION ────────────────────────────────────────────────────────
# StarDist saves automatically during training — no explicit save call needed.
# NOTE: model.export_TF() is broken with Keras 3+ (raises NotImplementedError).
# If you need a TF1 SavedModel for Fiji/ImageJ, export from a separate
# environment with tensorflow 1.x. See:
#   https://gist.github.com/uschmidt83/4b747862fe307044c722d6d1009f6183

final_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
if os.path.exists(final_path):
    saved_files = os.listdir(final_path)
    print(f"\n✓ Fine-tuned model saved to: {final_path}")
    print(f"  Files: {saved_files}")
else:
    print(f"\n⚠ Expected save folder not found at: {final_path}")
    print("  Check that training completed without errors.")


"""
To reload in NucleiStarDist.py (use the absolute path printed above):

    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    model = StarDist2D(None, name='stardist_finetuned', basedir=r'C:/your/path/to/models')
    img   = normalize(imread('my_image.png'), 1, 99.8)
    labels, details = model.predict_instances(img)
"""