import os

# Set your directory path
directory = "C:\\Users\\jonat\\Myelination\\SampleAugmented\\3"  # Raw string to handle backslashes

for filename in os.listdir(directory):
    if filename.endswith(".png") and not filename.startswith("Sample"):
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, f"Sample{filename}")
        
        # Rename the file (only if it doesn't already start with "Sample")
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → Sample{filename}")

print("All files renamed successfully!")