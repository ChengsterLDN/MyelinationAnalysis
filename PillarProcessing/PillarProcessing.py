#!/usr/bin/env python3
"""
LIF File Processor - Easy-to-use application for extracting C=2 channels from .lif files
"""

import os
import sys
import subprocess
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from pathlib import Path

# Global configuration
CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from file"""
    default_config = {
        "fiji_path": "",
        "last_directory": "",
        "default_channel": 2
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**default_config, **json.load(f)}
        except:
            return default_config
    return default_config

def save_config(config):
    """Save configuration to file"""
    config_dir = os.path.dirname(CONFIG_FILE)
    if config_dir:  # Only create directory if there's actually a directory path
        os.makedirs(config_dir, exist_ok=True)
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def find_fiji_suggestions():
    """Provide suggested paths for Fiji"""
    suggestions = []
    
    if sys.platform == "darwin":  # macOS
        suggestions = [
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
            os.path.expanduser("~/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx")
        ]
    elif sys.platform == "win32":  # Windows
        suggestions = [
            "C:\\Program Files\\Fiji.app\\ImageJ-win64.exe",
            "C:\\Fiji\\ImageJ-win64.exe",
            os.path.join(os.environ.get('PROGRAMFILES', ''), "Fiji.app", "ImageJ-win64.exe"),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), "Fiji.app", "ImageJ-win64.exe")
        ]
    else:  # Linux
        suggestions = [
            "/opt/Fiji.app/ImageJ-linux64",
            os.path.expanduser("~/Fiji.app/ImageJ-linux64"),
            "/usr/local/Fiji.app/ImageJ-linux64"
        ]
    
    # Filter to only existing paths
    return [path for path in suggestions if os.path.exists(path)]

def validate_fiji_path(path):
    """Validate if the provided path is a valid Fiji executable"""
    if not path or not os.path.exists(path):
        return False
    
    # Check if it's a Fiji executable
    executable_name = os.path.basename(path).lower()
    
    # Acceptable Fiji executables
    valid_executables = [
        'imagej-win64.exe', 
        'fiji.exe',
        'imagej.exe',
        'fiji-windows-x64.exe'  # Your specific case
    ]
    
    return any(valid_exe == executable_name for valid_exe in valid_executables)

def create_macro(lif_file, output_dir, channel=2):
    """Create the Fiji macro dynamically"""
    macro_content = f"""
// LIF Processor Macro - Channel {channel} Extractor
print("Processing LIF file: {lif_file}");

open("{lif_file.replace('"', '\\"')}");
n = nImages;
processed = 0;

for (i=0; i<n; i++) {{
    selectImage(i+1);
    title = getTitle();
    
    // Check if image has multiple channels
    if (nChannels > 1) {{
        // Select specified channel (1-based indexing)
        run("Channels Tool...");
        setChannel({channel});
        run("RGB Color");
        
        // Create output filename
        baseName = File.nameWithoutExtension;
        seriesName = replace(title, " ", "_");
        outputPath = "{output_dir.replace('"', '\\"')}" + File.separator + baseName + "_" + seriesName + "_C{channel}.png";
        
        // Save as PNG
        saveAs("PNG", outputPath);
        print("++ Saved: " + outputPath);
        processed++;
    }} else {{
        print("-- Skipping: " + title + " (only 1 channel)");
    }}
}}

close("*");
print("Processing complete! " + processed + " images exported.");
"""
    return macro_content

def test_fiji_installation(fiji_path):
    """Test if Fiji can be launched successfully"""
    try:
        cmd = [fiji_path, "--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"Fiji test command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Test error: {e}")
        return False
    
def process_lif_file(lif_path, output_dir=None, channel=2, fiji_path=None):
    """Process a single LIF file"""
    config = load_config()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(lif_path), "Exported_C2")
    
    # Use provided Fiji path or config path
    fiji_path = fiji_path or config.get('fiji_path', '')
    
    if not fiji_path:
        raise ValueError("Fiji path not specified. Please provide the path to Fiji.")
    
    if not validate_fiji_path(fiji_path):
        raise ValueError(f"Invalid Fiji path: {fiji_path}. Please provide the correct path to Fiji executable.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary macro file with absolute path
    macro_content = create_macro(lif_path, output_dir, channel)
    macro_file = os.path.join(output_dir, "temp_process_lif.ijm")
    
    with open(macro_file, 'w', encoding='utf-8') as f:
        f.write(macro_content)
    
    # Get absolute path to macro file
    macro_file_abs = os.path.abspath(macro_file)
    
    try:
        # Use the FULL PATH to Fiji executable (don't change directory)
        cmd = [fiji_path, "--headless", "-macro", macro_file_abs]
        
        print(f"Processing {lif_path}...")
        print(f"Output directory: {output_dir}")
        print(f"Using Fiji: {fiji_path}")
        print("This may take a few minutes depending on file size...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Print Fiji output
        if result.stdout:
            print("Fiji output:")
            for line in result.stdout.split('\n'):
                if line.strip() and ('++' in line or '--' in line or 'Processing' in line):
                    print(line)
        
        if result.returncode == 0:
            print(f"Successfully processed {lif_path}")
            # Save successful path to config
            config['fiji_path'] = fiji_path
            config['last_directory'] = os.path.dirname(lif_path)
            save_config(config)
            return True
        else:
            print(f"Error processing {lif_path}")
            if result.stderr:
                print("Error details:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Processing timed out. The file might be very large.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up macro file
        if os.path.exists(macro_file):
            os.remove(macro_file)

class FijiPathDialog:
    """Dialog to ask for Fiji path"""
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Fiji Path Required")
        self.dialog.geometry("600x300")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        # Main message
        tk.Label(self.dialog, text="Fiji/ImageJ Path Required", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Explanation
        explanation = """Please locate your Fiji installation.

Fiji is required to process .lif files. If you don't have Fiji installed, 
download it from: https://fiji.sc/

On Windows: Look for ImageJ-win64.exe
On Mac: Look for ImageJ-macosx in Fiji.app/Contents/MacOS/
On Linux: Look for ImageJ-linux64
"""
        tk.Label(self.dialog, text=explanation, justify=tk.LEFT, wraplength=550).pack(pady=5, padx=20)
        
        # Path entry frame
        path_frame = tk.Frame(self.dialog)
        path_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.path_var = tk.StringVar()
        tk.Entry(path_frame, textvariable=self.path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(path_frame, text="Browse", command=self.browse_fiji).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Suggested paths
        suggestions = find_fiji_suggestions()
        if suggestions:
            suggestion_frame = tk.Frame(self.dialog)
            suggestion_frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(suggestion_frame, text="Common locations:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
            
            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
                btn = tk.Button(suggestion_frame, text=f"Use: {suggestion}", 
                               command=lambda s=suggestion: self.path_var.set(s),
                               relief=tk.FLAT, fg="blue", cursor="hand2")
                btn.pack(anchor=tk.W, pady=2)
        
        # Button frame
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="OK", command=self.ok, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # Center dialog
        self.dialog.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
    def browse_fiji(self):
        """Browse for Fiji executable"""
        if sys.platform == "win32":
            filetypes = [("Fiji Executable", "*.exe"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]
        
        path = filedialog.askopenfilename(
            title="Select Fiji/ImageJ Executable",
            filetypes=filetypes
        )
        if path:
            self.path_var.set(path)
    
    def ok(self):
        """OK button handler"""
        path = self.path_var.get().strip()
        if path and validate_fiji_path(path):
            self.result = path
            self.dialog.destroy()
        else:
            messagebox.showerror("Invalid Path", "Please select a valid Fiji/ImageJ executable.")
    
    def cancel(self):
        """Cancel button handler"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and return result"""
        self.parent.wait_window(self.dialog)
        return self.result

def gui_mode():
    """Graphical user interface mode"""
    root = tk.Tk()
    root.title("LIF File Processor")
    root.geometry("600x400")
    
    config = load_config()
    
    def select_lif_file():
        initial_dir = config.get('last_directory', '')
        file_path = filedialog.askopenfilename(
            title="Select .lif file",
            filetypes=[("LIF files", "*.lif"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if file_path:
            lif_var.set(file_path)
            config['last_directory'] = os.path.dirname(file_path)
            save_config(config)
    
    def select_output_dir():
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            output_var.set(dir_path)
    
    def get_fiji_path():
        """Get Fiji path, asking user if not configured"""
        fiji_path = config.get('fiji_path', '')
        if not fiji_path or not validate_fiji_path(fiji_path):
            dialog = FijiPathDialog(root)
            fiji_path = dialog.show()
            if fiji_path:
                config['fiji_path'] = fiji_path
                save_config(config)
                return fiji_path
            else:
                return None
        return fiji_path
    
    def process_file():
        lif_file = lif_var.get()
        output_dir = output_var.get() or os.path.join(os.path.dirname(lif_file), "Exported_C2")
        
        if not lif_file:
            messagebox.showerror("Error", "Please select a .lif file")
            return
        
        fiji_path = get_fiji_path()
        if not fiji_path:
            return  # User cancelled Fiji path selection
        
        try:
            if process_lif_file(lif_file, output_dir, channel_var.get(), fiji_path):
                messagebox.showinfo("Success", f"Processing complete!\nFiles saved to: {output_dir}")
            else:
                messagebox.showerror("Error", "Processing failed. Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")
        
        fiji_path = get_fiji_path()
        if not fiji_path:
            return
    
        # Test Fiji first
        if not test_fiji_installation(fiji_path):
            messagebox.showerror("Error", "Fiji cannot be launched. Please check the installation.")
            return
    
    # GUI elements
    tk.Label(root, text="LIF File Processor", font=("Arial", 16, "bold")).pack(pady=10)
    
    lif_var = tk.StringVar()
    output_var = tk.StringVar()
    channel_var = tk.IntVar(value=config.get('default_channel', 2))
    
    # Main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # LIF file selection
    tk.Label(main_frame, text="Select .lif file:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(0, 5))
    frame1 = tk.Frame(main_frame)
    frame1.pack(fill='x', pady=(0, 10))
    tk.Entry(frame1, textvariable=lif_var, width=60).pack(side='left', fill='x', expand=True)
    tk.Button(frame1, text="Browse", command=select_lif_file).pack(side='right')
    
    # Output directory selection
    tk.Label(main_frame, text="Output directory (optional):", font=("Arial", 10, "bold")).pack(anchor='w', pady=(0, 5))
    frame2 = tk.Frame(main_frame)
    frame2.pack(fill='x', pady=(0, 10))
    tk.Entry(frame2, textvariable=output_var, width=60).pack(side='left', fill='x', expand=True)
    tk.Button(frame2, text="Browse", command=select_output_dir).pack(side='right')
    
    # Channel selection
    channel_frame = tk.Frame(main_frame)
    channel_frame.pack(fill='x', pady=(0, 20))
    tk.Label(channel_frame, text="Channel to extract:", font=("Arial", 10, "bold")).pack(side='left')
    tk.Spinbox(channel_frame, from_=1, to=10, width=5, textvariable=channel_var).pack(side='left', padx=(5, 0))
    
    # Process button
    tk.Button(main_frame, text="Process LIF File", command=process_file, 
             bg='#4CAF50', fg='white', font=("Arial", 12, "bold"), padx=20, pady=10).pack(pady=20)
    
    # Status/info
    info_text = """Instructions:
1. Select a .lif file to process
2. Choose output directory (optional)
3. Select which channel to extract
4. Click 'Process LIF File'
5. First time users will be asked to locate Fiji installation

Note: Requires Fiji/ImageJ to be installed
Download from: https://fiji.sc/"""
    
    tk.Label(main_frame, text=info_text, justify=tk.LEFT, font=("Arial", 9)).pack(side='bottom', anchor='w', pady=10)
    
    # Show current Fiji path if configured
    fiji_path = config.get('fiji_path', '')
    if fiji_path and validate_fiji_path(fiji_path):
        status_text = f"✓ Fiji configured: {fiji_path}"
        tk.Label(main_frame, text=status_text, fg="green", font=("Arial", 9)).pack(side='bottom', anchor='w', pady=5)
    
    root.mainloop()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process LIF files and export specific channels as PNG')
    parser.add_argument('file', nargs='?', help='Path to .lif file')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-c', '--channel', type=int, default=2, help='Channel to extract (default: 2)')
    parser.add_argument('--fiji-path', help='Path to Fiji executable')
    parser.add_argument('--gui', action='store_true', help='Launch graphical interface')
    
    args = parser.parse_args()
    
    if args.gui or not args.file:
        gui_mode()
    else:
        config = load_config()
        fiji_path = args.fiji_path or config.get('fiji_path', '')
        
        if not fiji_path:
            print("Error: Fiji path not specified.")
            print("Please provide Fiji path using --fiji-path argument")
            print("Or run with --gui to configure interactively")
            sys.exit(1)
        
        if not validate_fiji_path(fiji_path):
            print(f"Error: Invalid Fiji path: {fiji_path}")
            sys.exit(1)
        
        process_lif_file(args.file, args.output, args.channel, fiji_path)

if __name__ == "__main__":
    main()