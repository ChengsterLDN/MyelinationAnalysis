import os
import json
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import subprocess
import tempfile
import time

def random_selection(parent_directory, percentage=10):
    """Select random series from the parent directory."""
    subfolders = []
    
    for item in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, item)
        if os.path.isdir(subfolder_path):
            # Check for required images
            mbp_path = os.path.join(subfolder_path, 'mbp_mip.png')
            pillar_path = os.path.join(subfolder_path, 'pillar_mip.png')
            
            if os.path.exists(mbp_path) and os.path.exists(pillar_path):
                subfolders.append({
                    'path': subfolder_path,
                    'name': item,
                    'mbp_path': mbp_path,
                    'pillar_path': pillar_path
                })
    
    num_to_select = max(1, int(len(subfolders) * percentage / 100))
    num_to_select = min(num_to_select, len(subfolders))
    selected = random.sample(subfolders, num_to_select)
    
    return selected

def batch_save(batch_content, output_file):
    """Saves the current state of scores to the JSON file."""
    with open(output_file, 'w') as f:
        json.dump(batch_content, f, indent=2)
    print(f"Updated results saved to: {output_file}")

def gui(batch_content, output_file):
    root = tk.Tk()
    root.title("Quality Control Class Scoring")
    root.geometry("800x600")
    
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tk.Label(frame, text="Randomly Selected Series", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(frame, text="Double-click a row to Launch ManualScore").pack()

    # Listbox to show folders and current scores
    listbox = tk.Listbox(frame, height=15, font=("Courier", 10))
    listbox.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Scrollbar for listbox
    scrollbar = tk.Scrollbar(listbox)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    def refresh_list():
        listbox.delete(0, tk.END)
        for i, item in enumerate(batch_content, 1):
            score = item.get('class_score', "Not Scored")
            scores_info = ""
            if 'scores' in item:
                scores = item['scores']
                scores_info = f" | 0:{scores.get('0', 0)} 1:{scores.get('1', 0)} 2:{scores.get('2', 0)} 3:{scores.get('3', 0)}"
            listbox.insert(tk.END, f"{i}. [{score}] - {item['subfolder_name']}{scores_info}")

    refresh_list()

    def score_item(event=None):
        selection = listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        item = batch_content[index]

        pillar_path = item['pillar_path']
        mbp_path = item['mbp_path']
        
        print(f"Launching ManualScore for: {item['subfolder_name']}")
        
        # Disable the button while scoring
        score_button.config(state=tk.DISABLED)
        status_label.config(text=f"Scoring: {item['subfolder_name']}...")
        root.update()
        
        try:
            # Create a unique temporary file for this scoring session
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"manualscore_{item['subfolder_name']}_{int(time.time())}.json")
            
            # Delete any existing temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            manualscore_path = os.path.join(script_dir, "ManualScore.py")
            
            if os.path.exists(manualscore_path):
                # Run ManualScore with the temp file as third argument
                process = subprocess.Popen(
                    [sys.executable, manualscore_path, pillar_path, mbp_path, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for the process to complete
                process.wait()
                
                # Check if temp file was created and contains data
                if os.path.exists(temp_file):
                    try:
                        with open(temp_file, 'r') as f:
                            scores_data = json.load(f)
                        
                        # Update batch content with scores
                        batch_content[index]['scores'] = scores_data['scores']
                        batch_content[index]['class_score'] = "Scored"
                        batch_content[index]['total_scored'] = scores_data['total_scored']
                        batch_content[index]['total_boxes'] = scores_data['total_boxes']
                        
                        batch_save(batch_content, output_file)
                        refresh_list()
                        
                        # Show success message
                        scores = scores_data['scores']
                        messagebox.showinfo(
                            "Scoring Complete", 
                            f"Scores saved for {item['subfolder_name']}:\n"
                            f"Score 0: {scores['0']}\n"
                            f"Score 1: {scores['1']}\n"
                            f"Score 2: {scores['2']}\n"
                            f"Score 3: {scores['3']}\n"
                            f"Total: {scores_data['total_scored']} boxes scored"
                        )
                        
                        # Clean up temp file
                        os.remove(temp_file)
                        
                    except json.JSONDecodeError:
                        messagebox.showerror("Error", f"Could not read scores from temporary file.")
                        # Clean up corrupted temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                else:
                    messagebox.showinfo("No Scores", "ManualScore was closed without saving scores.")
                    
            else:
                messagebox.showerror("Error", f"Could not find ManualScore.py at {manualscore_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run ManualScore:\n{e}")
        finally:
            # Re-enable the button
            score_button.config(state=tk.NORMAL)
            status_label.config(text="")
    
    def finish_and_close():
        """Close the application properly."""
        # Save final state
        batch_save(batch_content, output_file)
        
        # Calculate summary
        scored_items = [item for item in batch_content if item.get('class_score') == 'Scored']
        scored_count = len(scored_items)
        
        # Prepare summary text
        summary_text = f"QUALITY CONTROL SUMMARY\n"
        summary_text += "="*50 + "\n"
        summary_text += f"Total samples: {len(batch_content)}\n"
        summary_text += f"Samples scored: {scored_count}\n"
        summary_text += f"Results saved to: {output_file}\n\n"
        
        if scored_count > 0:
            # Calculate totals and averages
            total_scores = {'0': 0, '1': 0, '2': 0, '3': 0}
            total_boxes = 0
            total_scored = 0
            
            summary_text += "INDIVIDUAL RESULTS:\n"
            summary_text += "-"*50 + "\n"
            
            for item in scored_items:
                scores = item.get('scores', {})
                total_scored_item = item.get('total_scored', 0)
                total_boxes_item = item.get('total_boxes', 0)
                
                summary_text += f"{item['subfolder_name']}:\n"
                summary_text += f"  Score 0: {scores.get('0', 0):3d} | Score 1: {scores.get('1', 0):3d} | "
                summary_text += f"Score 2: {scores.get('2', 0):3d} | Score 3: {scores.get('3', 0):3d}\n"
                summary_text += f"  Scored: {total_scored_item}/{total_boxes_item} boxes\n"
                
                # Add to totals
                for key in total_scores:
                    total_scores[key] += scores.get(key, 0)
                total_scored += total_scored_item
                total_boxes += total_boxes_item
            
            summary_text += "\n" + "="*50 + "\n"
            summary_text += "TOTALS:\n"
            summary_text += f"Score 0: {total_scores['0']:5d}\n"
            summary_text += f"Score 1: {total_scores['1']:5d}\n"
            summary_text += f"Score 2: {total_scores['2']:5d}\n"
            summary_text += f"Score 3: {total_scores['3']:5d}\n"
            summary_text += f"Total boxes scored: {total_scored}\n"
            summary_text += f"Total boxes available: {total_boxes}\n"
            
            # Calculate percentages
            if total_scored > 0:
                summary_text += "\nPERCENTAGES:\n"
                for key in ['0', '1', '2', '3']:
                    percentage = (total_scores[key] / total_scored) * 100
                    summary_text += f"Score {key}: {percentage:6.1f}%\n"
        
        # Show summary in messagebox and also save to text file
        messagebox.showinfo("Summary", summary_text)
        
        # Also save summary to a text file
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        print(f"Summary saved to: {summary_file}")
        
        # Close the GUI and exit program
        root.destroy()
        sys.exit(0)

    listbox.bind('<Double-1>', score_item)

    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    score_button = tk.Button(button_frame, text="Score Selected", command=score_item, bg="blue", fg="white")
    score_button.pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Finish & Close", command=finish_and_close, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
    
    # Status label
    status_label = tk.Label(frame, text="", fg="blue")
    status_label.pack(pady=5)
    
    # Instructions
    tk.Label(frame, text="Instructions:", font=("Arial", 10, "bold")).pack(pady=(20, 5))
    tk.Label(frame, text="1. Double-click a sample to open ManualScore", font=("Arial", 9)).pack(anchor="w")
    tk.Label(frame, text="2. Click 'Detect Pillar Centers' or add dots manually", font=("Arial", 9)).pack(anchor="w")
    tk.Label(frame, text="3. Click 'Create Boxes' to generate scoring boxes", font=("Arial", 9)).pack(anchor="w")
    tk.Label(frame, text="4. Score boxes (0-3) and click 'Save & Close'", font=("Arial", 9)).pack(anchor="w")
    tk.Label(frame, text="5. Scores will be automatically saved", font=("Arial", 9)).pack(anchor="w")
    
    root.mainloop()

def main():
    print("="*60)
    print("QUALITY CONTROL SCORING TOOL")
    print("="*60)
    
    root = tk.Tk()
    root.withdraw()
    
    parent_directory = filedialog.askdirectory(title="Select Parent Directory")
    if not parent_directory:
        return
    
    # Check if a scoring file already exists to resume work
    batch_file = os.path.join(parent_directory, "manual_scores.json")
    
    if os.path.exists(batch_file):
        use_existing = messagebox.askyesno("Resume?", "A previous scoring file (manual_scores.json) was found.\nDo you want to load it?")
        if use_existing:
            with open(batch_file, 'r') as f:
                batch_content = json.load(f)
            print(f"Loaded existing batch file with {len(batch_content)} entries.")
            gui(batch_content, batch_file)
            return

    # If not resuming, create new selection
    selected_subfolders = random_selection(parent_directory, percentage=10)
    
    if not selected_subfolders:
        messagebox.showinfo("No Subfolders", "No valid subfolders found.")
        return
    
    batch_content = []
    for subfolder in selected_subfolders:
        batch_content.append({
            'subfolder_name': subfolder['name'],
            'subfolder_path': subfolder['path'],
            'class_score': None,
            'pillar_path': subfolder['pillar_path'],
            'mbp_path': subfolder['mbp_path']
        })
    
    # Save initial file
    batch_save(batch_content, batch_file)
    
    messagebox.showinfo("Selection Complete", 
                        f"Selected {len(batch_content)} series.\nResults will save to manual_scores.json")
    
    # Start the scoring GUI
    gui(batch_content, batch_file)
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print(f"Final scores saved in: {batch_file}")
    print("="*60)

if __name__ == "__main__":
    main()