import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Optional
from skimage import filters, morphology
import cv2

class MIPProcessor:
    """Maximum Intensity Projection processing"""
    
    def __init__(self):

        self.supported_formats = {'.png', '.jpg', '.jpeg'}
    
    def create_mip(self, image_paths: List[str], dim: bool = False, 
                   apply_otsu: bool = False, apply_yellow: bool = False) -> Optional[Image.Image]:
        """
        Args:
            image_paths: List of paths to images in the stack
            dim_to_25_percent: Whether to dim each image to 25% brightness
            apply_otsu: Whether to apply Otsu thresholding
            apply_yellow: Whether to apply yellow mask to the MIP
            
        Returns:
            PIL Image object containing the MIP, or None if processing fails
        """
        if not image_paths:
            return None
        
        try:
            # Load first image to get dimensions
            first_img = Image.open(image_paths[0])
            width, height = first_img.size
            
            # Initialise MIP array
            mip_array = np.zeros((height, width, 3), dtype=np.float32)
            
            for img_path in image_paths:
                # Load and process each image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32)
                
                # Apply 25% dimming only if requested
                if dim:
                    img_array = img_array * 0.25
                
                # Update MIP - take maximum intensity at each pixel
                mip_array = np.maximum(mip_array, img_array)
            
            # Convert back to PIL Image
            mip_array = np.clip(mip_array, 0, 255).astype(np.uint8)
            mip_image = Image.fromarray(mip_array)
            
            # Apply Otsu thresholding if requested
            if apply_otsu:
                mip_image = self.apply_otsu(mip_image)
            
            # Apply yellow mask if requested
            if apply_yellow:
                mip_image = self.apply_yellow(mip_image)
            
            return mip_image
            
        except Exception as e:
            print(f"Error creating MIP: {e}")
            return None

    def apply_otsu(self, image: Image.Image) -> Image.Image:
        """
        Apply Otsu thresholding to an image with denoising.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Thresholded PIL Image
        """
        try:
            # Convert to grayscale for thresholding
            gray_image = image.convert('L')
            gray_array = np.array(gray_image)
            
            # Calculate Otsu threshold
            threshold = filters.threshold_otsu(gray_array)
            
            # Apply threshold
            binary_array = (gray_array > threshold).astype(np.uint8) * 255
            
            # Apply fastNlMeansDenoising
            binary_cleaned = cv2.fastNlMeansDenoising(binary_array)
            
            # Apply morphological denoising
            binary_final = self.apply_morphological_denoising(binary_cleaned)
            
            # Convert back to RGB
            binary_rgb = np.stack([binary_final] * 3, axis=-1)
            
            return Image.fromarray(binary_rgb)
            
        except Exception as e:
            print(f"Error applying Otsu threshold: {e}")
            return image

    def apply_morphological_denoising(self, binary_array: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to denoise binary image.
        
        Args:
            binary_array: Binary image array
            
        Returns:
            Denoised binary image array
        """
        try:
            # Remove small noise with opening
            kernel = np.ones((3, 3), np.uint8)
            denoised = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel)
            
            # Fill small holes with closing
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Remove very small objects
            denoised = morphology.remove_small_objects(denoised.astype(bool), min_size=50)
            denoised = denoised.astype(np.uint8) * 255
            
            return denoised
            
        except Exception as e:
            print(f"Error applying morphological denoising: {e}")
            return binary_array
    
    def apply_yellow(self, image: Image.Image) -> Image.Image:
        """
        Apply yellow mask to an image - only where there is white after denoising.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Yellow-masked PIL Image
        """
        try:
            # Convert image to array and apply Otsu thresholding with full denoising
            gray_image = image.convert('L')
            gray_array = np.array(gray_image)
            
            # Calculate Otsu threshold
            threshold = filters.threshold_otsu(gray_array)
            
            # Apply threshold
            binary_array = (gray_array > threshold).astype(np.uint8) * 255
            
            # Apply fastNlMeansDenoising
            binary_cleaned = cv2.fastNlMeansDenoising(binary_array)
            
            # Apply morphological denoising
            white_regions = self.apply_morphological_denoising(binary_cleaned)
            
            # Create yellow mask only where there are white regions
            img_array = np.array(image)
            yellow_mask = np.zeros_like(img_array)
            yellow_mask[:, :, 0] = 255  # Red channel
            yellow_mask[:, :, 1] = 255  # Green channel
            yellow_mask[:, :, 2] = 0    # Blue channel
            
            # Create mask for white regions (where final denoised binary is white)
            white_mask = white_regions > 0
            
            # Apply yellow only to white regions, keep original elsewhere
            result = img_array.copy()
            result[white_mask] = yellow_mask[white_mask]
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"Error applying yellow mask: {e}")
            return image

class FolderProcessor:
    """Processes folder structure and manages MIP creation"""
    
    def __init__(self, parent_folder: str):

        self.parent_folder = Path(parent_folder)
        self.mip_processor = MIPProcessor()
        self.target_folders = ['nuclei', 'mbp', 'pillar']
    
    def find_series_folders(self) -> List[Path]:

        series_folders = []
        for item in self.parent_folder.iterdir():
            if item.is_dir():
                series_folders.append(item)
        
        return sorted(series_folders, key=lambda x: x.name)
    
    def get_folder_images(self, folder_path: Path) -> List[str]:

        image_paths = []
        for img_file in folder_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in self.mip_processor.supported_formats:
                image_paths.append(str(img_file))
        
        return sorted(image_paths)
    
    def has_valid_pillar_images(self, folder_path: Path) -> bool:

        valid_folder = folder_path / 'valid'
        if valid_folder.exists():
            valid_images = self.get_folder_images(valid_folder)
            if valid_images:
                return True
        return False

    def get_valid_folder_images(self, folder_path: Path) -> List[str]:
        
        valid_folder = folder_path / 'valid'
        invalid_folder = folder_path / 'invalid'
        
        # First try to get images from valid folder
        if valid_folder.exists():
            valid_images = self.get_folder_images(valid_folder)
            if valid_images:
                return valid_images
        
        # If no valid images found, try invalid folder
        if invalid_folder.exists():
            invalid_images = self.get_folder_images(invalid_folder)
            if invalid_images:
                print(f"  No valid images found, using invalid folder for {folder_path.name}")
                return invalid_images
        
        # If neither valid nor invalid has images, try main folder as last resort
        main_images = self.get_folder_images(folder_path)
        if main_images:
            print(f"  No valid/invalid images found, using main folder for {folder_path.name}")
            return main_images
        
        return []
    
    def get_nuclei_images(self, nuclei_folder: Path) -> List[str]:

        return self.get_folder_images(nuclei_folder)
    
    def process_series_folder(self, series_folder: Path) -> bool:
       
        print(f"Processing folder: {series_folder.name}")
        
        # Check if pillar has valid images, skip if not
        pillar_folder = series_folder / 'pillar'
        if pillar_folder.exists():
            if not self.has_valid_pillar_images(pillar_folder):
                print(f"  Skipping {series_folder.name}: No valid pillar images found")
                return False
        else:
            print(f"  Warning: pillar folder not found in {series_folder.name}")
            return False
        
        success = True
        
        for folder_name in self.target_folders:
            target_folder = series_folder / folder_name
            if not target_folder.exists():
                print(f"  Warning: {folder_name} folder not found in {series_folder.name}")
                success = False
                continue
            
            # Get image paths based on folder type
            if folder_name == 'nuclei':
                image_paths = self.get_nuclei_images(target_folder)
            else:
                image_paths = self.get_valid_folder_images(target_folder)
            
            if not image_paths:
                print(f"  Warning: No images found in {folder_name}")
                success = False
                continue
            
            # Create MIP with appropriate processing
            if folder_name == 'nuclei':
                mip_image = self.mip_processor.create_mip(
                    image_paths, 
                    dim=True,
                    apply_otsu=True,
                    apply_yellow_mask=False
                )
            elif folder_name == 'pillar':
                mip_image = self.mip_processor.create_mip(
                    image_paths, 
                    dim=False,
                    apply_otsu=True,
                    apply_yellow_mask=True
                )
            else:  # mbp
                mip_image = self.mip_processor.create_mip(
                    image_paths, 
                    dim=False,
                    apply_otsu=False,
                    apply_yellow_mask=False
                )
            
            if mip_image is None:
                print(f"  Error: Failed to create MIP for {folder_name}")
                success = False
                continue
            
            # Save MIP in series folder
            output_path = series_folder / f"{folder_name}_mip.png"
            try:
                mip_image.save(output_path, 'PNG')
                
                # Create status message
                status_parts = []
                if folder_name == 'nuclei':
                    status_parts.extend(["25% dimmed", "Otsu + Denoising + Morphological"])
                elif folder_name == 'pillar':
                    status_parts.extend(["full brightness", "Otsu + Denoising + Morphological", "yellow mask"])
                else:  # mbp
                    status_parts.append("full brightness")
                
                status_str = " + ".join(status_parts)
                print(f"  Created MIP: {output_path.name} ({status_str})")
                
            except Exception as e:
                print(f"  Error saving MIP for {folder_name}: {e}")
                success = False
        
        return success
    
    def process_all_series(self) -> dict:
        """
        Process all folders in the parent folder.
        
        Returns:
            Dictionary with processing results
        """
        series_folders = self.find_series_folders()
        
        if not series_folders:
            return {
                'success': False, 
                'message': 'No subfolders found',
                'total_folders': 0,
                'successful_folders': 0,
                'failed_folders': 0,
                'details': []
            }
        
        results = {
            'total_folders': len(series_folders),
            'successful_folders': 0,
            'failed_folders': 0,
            'details': []
        }
        
        for series_folder in series_folders:
            if self.process_series_folder(series_folder):
                results['successful_folders'] += 1
                results['details'].append(f"✓ {series_folder.name} - Success")
            else:
                results['failed_folders'] += 1
                results['details'].append(f"✗ {series_folder.name} - Failed")
        
        results['success'] = results['failed_folders'] == 0
        return results

def main():
    """Main function to run the MIP processing."""
    # Use filedialog to select folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("Select the parent folder containing your data folders")
    parent_folder = filedialog.askdirectory(title="Select Parent Folder")
    
    if not parent_folder:
        print("No folder selected. Exiting.")
        return
    
    print("\nProcessing details:")
    print("- Nuclei: 25% dimmed + Otsu thresholding + Denoising + Morphological")
    print("- MBP: Full brightness (no additional processing)")
    print("- Pillar: Full brightness + Otsu thresholding + Denoising + Morphological + Yellow mask")
    print("- Folder priority: valid → invalid → main folder")
    print("- Series folders skipped if no valid pillar images found")
    
    # Process folders
    processor = FolderProcessor(parent_folder)
    results = processor.process_all_series()
    
    # Print results
    print(f"\n{'='*50}")
    print("PROCESSING RESULTS:")
    print(f"{'='*50}")
    
    if 'message' in results:
        print(f"Error: {results['message']}")
    else:
        print(f"Total folders processed: {results['total_folders']}")
        print(f"Successful: {results['successful_folders']}")
        print(f"Failed: {results['failed_folders']}")
        
        if results['details']:
            print(f"\nDetails:")
            for detail in results['details']:
                print(f"  {detail}")
        
        if results['success']:
            print(f"\n✓ All MIPs created successfully!")
        else:
            print(f"\n⚠ Processing completed with some errors")
    
    print(f"\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()