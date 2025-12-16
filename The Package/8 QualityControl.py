import os
import json
import random
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

def random_selection(parent_directory, percentage=10):
    """
    Select random series from the parent directory.
    
    args:
        parent_directory: Path to the parent directory containing subfolders
        percentage: Percentage of subfolders to select (we'll default to 10% - might change to 5% idk)
    """
    subfolders = []
    
    for item in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, item)
        if os.path.isdir(subfolder_path):
            # Check for required images (NO NAME FILTERING)
            mbp_path = os.path.join(subfolder_path, 'mbp_mip.png')
            pillar_path = os.path.join(subfolder_path, 'pillar_mip.png')
            
            if os.path.exists(mbp_path) and os.path.exists(pillar_path):
                subfolders.append({
                    'path': subfolder_path,
                    'name': item,
                    'mbp_path': mbp_path,
                    'pillar_path': pillar_path
                })
    
    print(f"Found {len(subfolders)} valid subfolders with mbp_mip.png and pillar_mip.png")
    
    # Calculate how many to select (minimum 1, maximum available)
    num_to_select = max(1, int(len(subfolders) * percentage / 100))
    num_to_select = min(num_to_select, len(subfolders))
    
    # Randomly select subfolders
    selected = random.sample(subfolders, num_to_select)
    
    print(f"Randomly selected {len(selected)} subfolders ({percentage}%):")
    for item in selected:
        print(f"  - {item['name']}")
    
    return selected

def batch(selected_subfolders, output_file="manual_scoring_batch.txt"):
    
    # probs just need the subfolder name to compare for any signfiicant differences...
    batch_content = []
    
    for subfolder in selected_subfolders:
        batch_content.append({
            'subfolder_name': subfolder['name'],
            'subfolder_path': subfolder['path'],
            'mbp_image': subfolder['mbp_path'],
            'pillar_image': subfolder['pillar_path']
        })
    
    with open(output_file, 'w') as f:
        json.dump(batch_content, f, indent=2)
    
    print(f"\nBatch file saved to: {output_file}")
    
    # outputs to terminal just in case
    txt_file = "manual_scoring_summary.txt"
    with open(txt_file, 'w') as f:
        f.write("Manual Scoring Batch - Selected Subfolders\n")
        f.write("=" * 50 + "\n\n")
        for item in batch_content:
            f.write(f"Subfolder: {item['subfolder_name']}\n")
            f.write(f"Path: {item['subfolder_path']}\n")
            f.write(f"MBP Image: {item['mbp_image']}\n")
            f.write(f"Pillar Image: {item['pillar_image']}\n")
            f.write("-" * 50 + "\n\n")
    
    return batch_content

def call_manual(mbp_path, pillar_path, subfolder_name):
    """Launch ManualScore.py with pre-loaded image paths"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        manual_score_script = os.path.join(current_dir, "ManualScore.py")
        
        if not os.path.exists(manual_score_script):
            manual_score_script = "ManualScore.py"
            
        if os.path.exists(manual_score_script):
            print(f"\nLaunching manual scoring for: {subfolder_name}")
            print(f"  Pillar image: {pillar_path}")
            print(f"  MBP image: {mbp_path}")
            
            # Single command string
            command = f'"{sys.executable}" "{manual_score_script}" "{pillar_path}" "{mbp_path}"'
            print(f"  Command: {command}")
            
            # Run as a single string with shell=True
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            
            if result.returncode != 0:
                print(f"ManualScore.py exited with error (code {result.returncode})")
                print(f"Error output: {result.stderr}")
                return False
            return True
            
        else:
            print(f"Warning: ManualScore.py not found at {manual_score_script}")
            return False
            
    except Exception as e:
        print(f"Error launching ManualScore.py: {e}")
        return False

def gui(batch_content):
    root = tk.Tk()
    root.title("Manual Scoring Launcher")
    root.geometry("600x400")
    
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    title_label = tk.Label(frame, text="Selected Subfolders for Manual Scoring", 
                          font=("Courier", 14, "bold"))
    title_label.pack(pady=10)

    subtitle = tk.Label(frame, text="Click 'Start Manual Scoring' to begin processing each subfolder one by one")
    subtitle.pack(pady=5)

    listbox = tk.Listbox(frame, height=10, font=("Courier", 10))
    listbox.pack(fill=tk.BOTH, expand=True, pady=10)
    
    for i, item in enumerate(batch_content, 1):
        listbox.insert(tk.END, f"{i}. {item['subfolder_name']}")
    
    # Status label
    status_label = tk.Label(frame, text=f"Total subfolders selected: {len(batch_content)}", 
                           font=("Courier", 10))
    status_label.pack(pady=5)
    
    def start_manual_scoring():
        """Start manual scoring process for all selected subfolders"""
        if not batch_content:
            messagebox.showinfo("No Subfolders", "No subfolders selected for manual scoring.")
            return
        
        response = messagebox.askyesno(
            "Start Manual Scoring",
            f"You are about to start manual scoring for {len(batch_content)} subfolders.\n\n"
            f"This will open the ManualScore.py application for each subfolder one by one.\n\n"
            f"The image paths will be automatically loaded - no need to select them manually.\n\n"
            f"Continue?"
        )
        
        if response:
            root.destroy()  # Close the launcher window
            
            print("\n" + "="*60)
            print("STARTING MANUAL SCORING PROCESS")
            print("="*60)
            print("Note: Image paths will be automatically loaded for each folder")
            print("="*60)
            
            for i, item in enumerate(batch_content, 1):
                print(f"\n[{i}/{len(batch_content)}] Processing: {item['subfolder_name']}")
                print(f"  Pillar image: {item['pillar_image']}")
                print(f"  MBP image: {item['mbp_image']}")
                
                # No need for user input - automatically launch
                print(f"\nLaunching ManualScore.py for {item['subfolder_name']}...")
                
                success = call_manual(item['mbp_image'], item['pillar_image'], item['subfolder_name'])
                
                if success:
                    print(f":) Completed manual scoring for {item['subfolder_name']}")
                else:
                    print(f"!!! Failed to complete {item['subfolder_name']}")
                
                print("-"*50)
                
                # Ask if user wants to continue after each folder
                if i < len(batch_content):
                    next_folder = batch_content[i]['subfolder_name']
                    response = input(f"\nPress Enter to continue to {next_folder} or 'q' to quit: ")
                    if response.lower() == 'q':
                        print("\nManual scoring stopped by user.")
                        break
            
            print("\n" + "="*60)
            print("MANUAL SCORING PROCESS COMPLETED")
            print("="*60)
            print(f"\nProcessed {i} out of {len(batch_content)} subfolders.")
            print("You can now review and score the images.")
            
            messagebox.showinfo(
                "Process Complete",
                f"Manual scoring completed for {i} out of {len(batch_content)} subfolders.\n\n"
                f"Results are saved in each folder's scoring directories."
            )
    
    # Button frame
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    # Start button
    start_button = tk.Button(button_frame, text="Start Manual Scoring", 
                            command=start_manual_scoring,
                            bg="green", fg="white",
                            font=("Arial", 12, "bold"),
                            padx=20, pady=10)
    start_button.pack(side=tk.LEFT, padx=10)
    
    # Exit button
    exit_button = tk.Button(button_frame, text="Exit", 
                           command=root.destroy,
                           bg="red", fg="white",
                           font=("Arial", 12),
                           padx=20, pady=10)
    exit_button.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()

def main():
    root = tk.Tk()
    root.withdraw()
    
    print("="*60)
    print("RANDOM SUBFOLDER SELECTOR FOR MANUAL SCORING")
    print("="*60)
    
    # Ask user for parent directory
    print("\nPlease select the parent directory containing your subfolders...")
    parent_directory = filedialog.askdirectory(title="Select Parent Directory")
    
    if not parent_directory:
        print("No directory selected. Exiting.")
        return
    
    # Select random subfolders (10%)
    selected_subfolders = random_selection(parent_directory, percentage=10)
    
    if not selected_subfolders:
        messagebox.showinfo("No Subfolders", "No valid subfolders found with mbp_mip.png and pillar_mip.png.")
        return
    
    # Create batch json file
    batch_file = os.path.join(parent_directory, "manual_scoring_batch.json")
    batch_content = batch(selected_subfolders, batch_file)
    
    # Ask user if they want to start manual scoring
    response = messagebox.askyesno(
        "Manual Scoring Ready",
        f"Selected {len(selected_subfolders)} subfolders for manual scoring.\n\n"
        f"Batch file saved to:\n{batch_file}\n\n"
        f"Image paths will be automatically loaded for each folder.\n\n"
        f"Do you want to start the manual scoring process now?"
    )
    
    if response:
        gui(batch_content)
    else:
        print("\nBatch file created. You can start manual scoring later by:")
        print(f"1. Reviewing the batch file: {batch_file}")
        print(f"2. Running ManualScore.py manually for each selected subfolder")
        print("\nSelected subfolders:")
        for item in batch_content:
            print(f"  - {item['subfolder_name']}")
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()