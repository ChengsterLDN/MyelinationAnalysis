from PIL import Image
import os

#input_dir = ["C:\\Users\\jonat\\Myelination\\dataset\\train\\0","C:\\Users\\jonat\\Myelination\\dataset\\train\\1","C:\\Users\\jonat\\Myelination\\dataset\\train\\2","C:\\Users\\jonat\\Myelination\\dataset\\train\\3"]
input_dir = ["C:\\Users\\jonat\\Documents\\GitHub\\MyelinationAnalysis\\MBPValidationDataset\\invalid","C:\\Users\\jonat\\Documents\\GitHub\\MyelinationAnalysis\\MBPValidationDataset\\valid"]
output_dir = input_dir

print("Rotating Path: ", input_dir)
confirm = input("Proceed? (y/n): ").strip().lower()
if confirm != 'y':
    print("Operation cancelled.")
    exit()

for i in input_dir:
    for filename in os.listdir(i):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(i, filename)
            img = Image.open(img_path)

            # --- ROTATE 90 DEGREES ---
            rotated = img.rotate(90, expand=True)  # `expand=True` prevents cropping
            rotated.save(os.path.join(i, f"r90_{filename}"))

            # --- ROTATE 180 DEGREES ---
            rotated = img.rotate(180, expand=True)  # `expand=True` prevents cropping
            rotated.save(os.path.join(i, f"r180_{filename}"))

            # --- ROTATE 270 DEGREES ---
            rotated = img.rotate(270, expand=True)  # `expand=True` prevents cropping
            rotated.save(os.path.join(i, f"r270_{filename}"))

            print(f"Rotated: {filename}")

    for filename in os.listdir(i):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(i, filename)
            img = Image.open(img_path)

            # --- FLIP HORIZONTALLY (left-right) ---
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(os.path.join(i, f"f{filename}"))

            print(f"Flipped: {filename}")


print("Completed")
