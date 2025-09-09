import imagej
import os
import subprocess
import time

def run_imagej_macro(macro_path, lif_file_path, output_dir):
    """
    Run ImageJ macro using PyImageJ
    """
    try:
        # Initialize ImageJ
        print("Starting ImageJ...")
        ij = imagej.init('sc.fiji:fiji:2.14.0')  # You can specify a different version
        
        print("ImageJ initialized successfully")
        print(f"ImageJ version: {ij.getVersion()}")
        
        # Set macro parameters
        macro_params = {
            "lif_path": lif_file_path,
            "output_dir": output_dir
        }
        
        # Run the macro
        print("Running macro...")
        result = ij.py.run_macro(macro_path, macro_params)
        
        print("Macro execution completed successfully")
        return True
        
    except Exception as e:
        print(f"Error running macro: {e}")
        return False
    finally:
        try:
            ij.dispose()
            print("ImageJ disposed")
        except:
            pass

def main():
    # Configuration
    macro_path = "C:/Users/jonat/Myelination/Export-as-individual-images.ijm"
    lif_file_path = "C:/Users/jonat/Documents/My Documents/MecBioMed/MyelinationProject/Benz/Benz.lif"
    output_directory = "extracted_channel2_stacks"
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Run the macro
    success = run_imagej_macro(macro_path, lif_file_path, output_directory)
    
    if success:
        print("Extraction completed successfully!")
    else:
        print("Extraction failed!")

if __name__ == "__main__":
    main()