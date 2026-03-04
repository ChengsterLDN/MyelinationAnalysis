import os
import shutil
from tkinter import Tk, filedialog, messagebox


def collect_nuclei_images(source_dir, dest_dir):
    """
    Walk all subfolders in source_dir, find nuclei_mip.png files,
    and copy them to dest_dir renamed as <folder_name>_nuclei.png
    """
    copied  = 0
    skipped = 0

    for folder_name in sorted(os.listdir(source_dir)):
        subfolder_path = os.path.join(source_dir, folder_name)

        if not os.path.isdir(subfolder_path):
            continue

        source_image = os.path.join(subfolder_path, "nuclei_mip.png")

        if not os.path.exists(source_image):
            print(f"Skipping {folder_name}: nuclei_mip.png not found")
            skipped += 1
            continue

        new_name = f"{folder_name}_nuclei.png"
        dest_path = os.path.join(dest_dir, new_name)

        # Avoid silently overwriting if two folders somehow produce the same name
        if os.path.exists(dest_path):
            print(f"⚠️  Destination already exists, skipping: {new_name}")
            skipped += 1
            continue

        shutil.copy2(source_image, dest_path)
        print(f"✓ Copied: {folder_name}/nuclei_mip.png  →  {new_name}")
        copied += 1

    return copied, skipped


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    source_dir = filedialog.askdirectory(title="Select Parent Directory Containing Subfolders")
    if not source_dir:
        messagebox.showerror("Error", "No source directory selected.")
        exit()

    dest_dir = filedialog.askdirectory(title="Select Destination Folder for Renamed Images")
    if not dest_dir:
        messagebox.showerror("Error", "No destination directory selected.")
        exit()

    # Safety check — prevent copying into itself
    if os.path.abspath(dest_dir) == os.path.abspath(source_dir):
        messagebox.showerror("Error", "Source and destination folders must be different.")
        exit()

    copied, skipped = collect_nuclei_images(source_dir, dest_dir)

    messagebox.showinfo(
        "Done",
        f"Copied:  {copied} image(s)\n"
        f"Skipped: {skipped} folder(s)\n\n"
        f"Output folder:\n{dest_dir}"
    )

    root.destroy()