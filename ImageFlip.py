from PIL import Image
import os

input_dir = "C:\\Users\\jonat\\Myelination\\Myelination\\dataset\\train\\0"
output_dir = input_dir

print("Path: ", input_dir)
confirm = input("Proceed? (y/n): ").strip().lower()
if confirm != 'y':
    print("Operation cancelled.")
    exit()

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # --- FLIP HORIZONTALLY (left-right) ---
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(os.path.join(output_dir, f"f{filename}"))

        print(f"Processed: {filename}")

print("Flipped")
