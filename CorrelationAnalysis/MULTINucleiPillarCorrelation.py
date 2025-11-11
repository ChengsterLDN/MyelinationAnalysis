import json
import math
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, messagebox
import os
import glob

class PillarNucleiAnalyser:
    def __init__(self):
        self.wrapped_pillars = []
        self.nuclei_properties = []
        self.analysis_results = []

    def load_all_data(self):
        """Load all nuclei properties and pillar coordinates from multiple folders"""
        root = Tk()
        root.withdraw()
        
        print("Please select the root directory containing all your data folders...")
        root_dir = filedialog.askdirectory(title="Select Root Directory with All Data Folders")
        if not root_dir:
            messagebox.showerror("Error", "Please select a root directory.")
            root.destroy()
            return False
        
        # Find all nuclei properties JSON files
        nuclei_files = glob.glob(os.path.join(root_dir, "**", "*nuclei_props.json"), recursive=True)
        print(f"Found {len(nuclei_files)} nuclei properties files")
        
        # Find all pillar coordinates JSON files
        pillar_files = glob.glob(os.path.join(root_dir, "**", "*wrapped_pillars.json"), recursive=True)
        print(f"Found {len(pillar_files)} pillar coordinates files")
        
        if not nuclei_files or not pillar_files:
            messagebox.showerror("Error", f"No data files found in the selected directory.\nNuclei files: {len(nuclei_files)}\nPillar files: {len(pillar_files)}")
            root.destroy()
            return False
        
        # Load all nuclei properties
        self.nuclei_properties = []
        nuclei_id_counter = 0
        for nuclei_file in nuclei_files:
            try:
                with open(nuclei_file, 'r') as f:
                    nuclei_data = json.load(f)
                    # Ensure unique IDs across all datasets
                    for nucleus in nuclei_data:
                        nucleus['original_id'] = nucleus.get('nuclei_id', 0)
                        nucleus['nuclei_id'] = nuclei_id_counter
                        nuclei_id_counter += 1
                    self.nuclei_properties.extend(nuclei_data)
                    print(f"Loaded {len(nuclei_data)} nuclei from {os.path.basename(nuclei_file)}")
            except Exception as e:
                print(f"Error loading {nuclei_file}: {e}")
        
        # Load all pillar coordinates
        self.wrapped_pillars = []
        pillar_id_counter = 0
        for pillar_file in pillar_files:
            try:
                with open(pillar_file, 'r') as f:
                    pillar_data = json.load(f)
                    # Ensure unique IDs across all datasets
                    for pillar in pillar_data:
                        pillar['original_id'] = pillar.get('cell_id', 0)
                        pillar['cell_id'] = pillar_id_counter
                        pillar_id_counter += 1
                    self.wrapped_pillars.extend(pillar_data)
                    print(f"Loaded {len(pillar_data)} pillars from {os.path.basename(pillar_file)}")
            except Exception as e:
                print(f"Error loading {pillar_file}: {e}")
        
        root.destroy()
        print(f"\nTotal loaded: {len(self.nuclei_properties)} nuclei and {len(self.wrapped_pillars)} pillars")
        return True
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1['x'] - point2['x_c'])**2 + (point1['y'] - point2['y_c'])**2)
    
    def analyse_space(self, search_radius=50):
        """Analyse within search radius for all merged data, with each nucleus assigned only to its closest pillar"""
        self.analysis_results = []

        # Initialize results for each pillar
        for pillar in self.wrapped_pillars:
            self.analysis_results.append({
                'pillar_id': pillar['cell_id'],
                'original_pillar_id': pillar.get('original_id', pillar['cell_id']),
                'pillar_coords': pillar['center_coordinates'],
                'prox_nuclei': []  # will store nuclei assigned to this pillar
            })

        # For each nucleus, find the closest pillar within search_radius
        for nucleus in self.nuclei_properties:
            min_distance = float('inf')
            closest_pillar_index = -1

            for idx, pillar in enumerate(self.wrapped_pillars):
                distance = self.calculate_distance(pillar['center_coordinates'], nucleus)
                if distance < min_distance:
                    min_distance = distance
                    closest_pillar_index = idx

            # If a pillar is found within search_radius, assign nucleus to it
            if min_distance <= search_radius:
                self.analysis_results[closest_pillar_index]['prox_nuclei'].append({
                    'nuclei_id': nucleus['nuclei_id'],
                    'distance': min_distance,
                    'area': nucleus['area'],
                    'circularity': nucleus.get('circularity', 0),
                    'original_nuclei_id': nucleus.get('original_id', nucleus['nuclei_id'])
                })

        # Finalize results: compute counts and averages
        for result in self.analysis_results:
            prox_nuclei = result['prox_nuclei']
            result['prox_nuclei_count'] = len(prox_nuclei)
            result['total_nuclei_area'] = sum(nuc['area'] for nuc in prox_nuclei) if prox_nuclei else 0
            result['average_distance'] = np.mean([nuc['distance'] for nuc in prox_nuclei]) if prox_nuclei else 0

        print(f"Completed unique-assignment spatial analysis for {len(self.wrapped_pillars)} pillars (merged data)")
        return self.analysis_results
    
    def create_merged_plots(self, output_dir=None):
        """Create two comprehensive plots from all merged data"""
        
        if not self.analysis_results:
            print("No analysis results available. Run analyse_space first.")
            return
        
        if output_dir is None:
            root = Tk()
            root.withdraw()
            output_dir = filedialog.askdirectory(title="Select Output Directory for Merged Plots")
            root.destroy()
            if not output_dir:
                print("No output directory selected.")
                return
        
        # Prepare data for plotting from ALL results
        distances = []
        nuclei_areas = []
        nuclei_counts = []
        
        for result in self.analysis_results:
            for nucleus in result['prox_nuclei']:
                distances.append(nucleus['distance'])
                nuclei_areas.append(nucleus['area'])
            
            # For count plot, we need one data point per pillar
            if result['prox_nuclei_count'] > 0:
                avg_distance = result['average_distance']
                nuclei_counts.append((avg_distance, result['prox_nuclei_count']))
        
        # Create the comprehensive plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distance vs Nuclei Area (All Data)
        if distances and nuclei_areas:
            ax1.scatter(distances, nuclei_areas, alpha=0.6, color='blue', s=30)
            ax1.set_xlabel('Distance from Wrapped Pillar (pixels)')
            ax1.set_ylabel('Nuclei Area (pixels)')
            ax1.set_title('Distance vs Nuclei Area\n(All Individual Nuclei - Merged Data)')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            if len(distances) > 1:
                z = np.polyfit(distances, nuclei_areas, 1)
                p = np.poly1d(z)
                ax1.plot(distances, p(distances), "r--", alpha=0.8, 
                        label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
                ax1.legend()
        
        # Plot 2: Distance vs Nuclei Count (All Data)
        if nuclei_counts:
            count_distances, counts = zip(*nuclei_counts)
            ax2.scatter(count_distances, counts, alpha=0.6, color='green', s=50)
            ax2.set_xlabel('Average Distance from Wrapped Pillar (pixels)')
            ax2.set_ylabel('Number of Nuclei')
            ax2.set_title('Distance vs Nuclei Count\n(All Wrapped Pillars - Merged Data)')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(count_distances) > 1:
                z = np.polyfit(count_distances, counts, 1)
                p = np.poly1d(z)
                ax2.plot(count_distances, p(count_distances), "r--", alpha=0.8,
                        label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
                ax2.legend()
        
        plt.tight_layout()
        
        # Save merged plots
        output_path1 = os.path.join(output_dir, "merged_distance_vs_nuclei_area.png")
        output_path2 = os.path.join(output_dir, "merged_distance_vs_nuclei_count.png")
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Merged plots saved to {output_dir}")
        
        plt.show()
    
    def summary(self):
        """Generate summary statistics for all merged data"""
        if not self.analysis_results:
            print("No analysis results available.")
            return
        
        total_nuclei_near_pillars = sum(result['prox_nuclei_count'] for result in self.analysis_results)
        pillars_with_nuclei = sum(1 for result in self.analysis_results if result['prox_nuclei_count'] > 0)
        
        all_distances = []
        all_areas = []
        
        for result in self.analysis_results:
            for nucleus in result['prox_nuclei']:
                all_distances.append(nucleus['distance'])
                all_areas.append(nucleus['area'])
        
        print("\n=== MERGED SPATIAL ANALYSIS SUMMARY ===")
        print(f"Total wrapped pillars analysed: {len(self.wrapped_pillars)}")
        print(f"Total nuclei available: {len(self.nuclei_properties)}")
        print(f"Pillars with nearby nuclei: {pillars_with_nuclei} ({pillars_with_nuclei/len(self.wrapped_pillars)*100:.1f}%)")
        print(f"Total nuclei within 100 pixels: {total_nuclei_near_pillars}")
        print(f"Average nuclei per pillar: {total_nuclei_near_pillars/len(self.wrapped_pillars):.2f}")
        
        if all_distances:
            print(f"\nDistance statistics (pixels):")
            print(f"  Average: {np.mean(all_distances):.2f}")
            print(f"  Median: {np.median(all_distances):.2f}")
            print(f"  Min: {np.min(all_distances):.2f}")
            print(f"  Max: {np.max(all_distances):.2f}")
        
        if all_areas:
            print(f"\nNuclei area statistics (pixels):")
            print(f"  Average: {np.mean(all_areas):.2f}")
            print(f"  Median: {np.median(all_areas):.2f}")
            print(f"  Min: {np.min(all_areas):.2f}")
            print(f"  Max: {np.max(all_areas):.2f}")       

    
    def run_merged_analysis(self, search_radius=50):
        """Run complete analysis on all merged data"""
        print("Starting Merged Pillar-Nuclei Spatial Analysis...")
        
        if not self.load_all_data():
            return
        
        print(f"\nAnalysing all nuclei within {search_radius} pixels of all wrapped pillars...")
        self.analyse_space(search_radius)
        
        self.summary()

        print("\nCreating merged plots...")
        self.create_merged_plots()
        
        print("\n=== MERGED ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    # Create analyser instance and run merged analysis
    analyser = PillarNucleiAnalyser()
    
    try:
        analyser.run_merged_analysis(search_radius=50)
    except Exception as e:
        # Show error message if something goes wrong
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
        root.destroy()