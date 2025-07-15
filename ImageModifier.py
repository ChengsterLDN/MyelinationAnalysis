from PIL import Image
import os

input_dir = "C:\\Users\\jonat\\Myelination\\dataset\\train\\3"
output_dir = input_dir

print("Rotating Path: ", input_dir)
confirm = input("Proceed? (y/n): ").strip().lower()
if confirm != 'y':
    print("Operation cancelled.")
    exit()

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # --- ROTATE 90 DEGREES ---
        rotated = img.rotate(90, expand=True)  # `expand=True` prevents cropping
        rotated.save(os.path.join(output_dir, f"r90_{filename}"))

        # --- ROTATE 180 DEGREES ---
        rotated = img.rotate(180, expand=True)  # `expand=True` prevents cropping
        rotated.save(os.path.join(output_dir, f"r180_{filename}"))

        # --- ROTATE 270 DEGREES ---
        rotated = img.rotate(270, expand=True)  # `expand=True` prevents cropping
        rotated.save(os.path.join(output_dir, f"r270_{filename}"))

        print(f"Rotated: {filename}")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # --- FLIP HORIZONTALLY (left-right) ---
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(os.path.join(output_dir, f"f{filename}"))

        print(f"Flipped: {filename}")


print("Completed")
