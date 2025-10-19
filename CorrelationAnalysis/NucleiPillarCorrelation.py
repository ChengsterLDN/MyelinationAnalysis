import json
import math
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, messagebox
import os

class PillarNucleiAnalyser:
    def __init__(self):
        self.wrapped_pillars = []
        self.nuclei_properties = []
        self.analysis_results = []

    def load_data(self):
        root = Tk()
        root.withdraw()

        print("Please select the wrapped pillars JSON file...")
        pillars_path = filedialog.askopenfilename(title="Select Wrapped Pillars JSON File", filetypes = [("JSON files", "*.json"), ("All files", "*.*")])
        if not pillars_path:
            messagebox.showerror("Error", "Please select a wrapped pillars JSON file.")
            root.destroy()
            return False
            
        with open(pillars_path, 'r') as f:
            self.wrapped_pillars = json.load(f)
        print(f"Loaded {len(self.wrapped_pillars)} wrapped pillars")
        
        # Load nuclei properties
        print("Please select the nuclei properties JSON file...")
        nuclei_path = filedialog.askopenfilename(
            title="Select Nuclei Properties JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not nuclei_path:
            messagebox.showerror("Error", "Please select a nuclei properties JSON file.")
            root.destroy()
            return False
            
        with open(nuclei_path, 'r') as f:
            self.nuclei_properties = json.load(f)
        print(f"Loaded {len(self.nuclei_properties)} nuclei")
        
        root.destroy()
        return True
    
    def calculate_distance(self, point1, point2):
        # Calculate Euclidean distance between two points
        return math.sqrt((point1['x'] - point2['x_c'])**2 + (point1['y'] - point2['y_c'])**2)
    
    def analyse_space(self, search_radius=100):
        # Analyse within search radisu
        self.analysis_results = []
        
        for pillar in self.wrapped_pillars:
            pillar_coords = pillar['center_coordinates']
            prox_nuclei = []
            
            for nucleus in self.nuclei_properties:
                distance = self.calculate_distance(pillar_coords, nucleus)
                
                if distance <= search_radius:
                    prox_nuclei.append({
                        'nuclei_id': nucleus['nuclei_id'],
                        'distance': distance,
                        'area': nucleus['area'],
                        'circularity': nucleus.get('circularity', 0)
                    })
            
            # Sort by distance
            prox_nuclei.sort(key=lambda x: x['distance'])
            
            self.analysis_results.append({
                'pillar_id': pillar['cell_id'],
                'pillar_coords': pillar_coords,
                'prox_nuclei_count': len(prox_nuclei),
                'prox_nuclei':prox_nuclei,
                'total_nuclei_area': sum(nuc['area'] for nuc in prox_nuclei),
                'average_distance': np.mean([nuc['distance'] for nuc in prox_nuclei]) if prox_nuclei else 0
            })
        
        print(f"Completed spatial analysis for {len(self.wrapped_pillars)} pillars")
        return self.analysis_results
    
    def create_plots(self, output_dir=None):
        
        if not self.analysis_results:
            print("No analysis results available. Run analyse_spaces first.")
            return
        
        if output_dir is None:
            root = Tk()
            root.withdraw()
            output_dir = filedialog.askdirectory(title="Select Output Directory for Plots")
            root.destroy()
            if not output_dir:
                print("No output directory selected.")
                return
        
        # Prepare data for plotting
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
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distance vs Nuclei Area 
        if distances and nuclei_areas:
            ax1.scatter(distances, nuclei_areas, alpha=0.6, color='blue', s=30)
            ax1.set_xlabel('Distance from Wrapped Pillar (pixels)')
            ax1.set_ylabel('Nuclei Area (pixels)')
            ax1.set_title('Distance vs Nuclei Area\n(Individual Nuclei)')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            if len(distances) > 1:
                z = np.polyfit(distances, nuclei_areas, 1)
                p = np.poly1d(z)
                ax1.plot(distances, p(distances), "r--", alpha=0.8, 
                        label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
                ax1.legend()
        
        # Plot 2: Distance vs Nuclei Count 
        if nuclei_counts:
            count_distances, counts = zip(*nuclei_counts)
            ax2.scatter(count_distances, counts, alpha=0.6, color='green', s=50)
            ax2.set_xlabel('Average Distance from Wrapped Pillar (pixels)')
            ax2.set_ylabel('Number of Nuclei')
            ax2.set_title('Distance vs Nuclei Count\n(Per Wrapped Pillar)')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(count_distances) > 1:
                z = np.polyfit(count_distances, counts, 1)
                p = np.poly1d(z)
                ax2.plot(count_distances, p(count_distances), "r--", alpha=0.8,
                        label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
                ax2.legend()
        
        plt.tight_layout()
        
        # Save plots
        output_path1 = os.path.join(output_dir, "distance_vs_nuclei_area.png")
        output_path2 = os.path.join(output_dir, "distance_vs_nuclei_count.png")
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {output_dir}")
        
        plt.show()
    
    def summary(self):
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
        
        print("\n=== SPATIAL ANALYSIS SUMMARY ===")
        print(f"Total wrapped pillars analysed: {len(self.wrapped_pillars)}")
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
    
    def save_detailed_results(self, output_dir=None):
        """Save detailed analysis results to JSON"""
        if not self.analysis_results:
            print("No analysis results available.")
            return
        
        if output_dir is None:
            root = Tk()
            root.withdraw()
            output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
            root.destroy()
            if not output_dir:
                print("No output directory selected.")
                return
        
        output_path = os.path.join(output_dir, "pillar_nuclei_analysis_detailed.json")
        
        # Create a simplified version for saving
        simplified_results = []
        for result in self.analysis_results:
            simplified_results.append({
                'pillar_id': result['pillar_id'],
                'pillar_coords': result['pillar_coords'],
                'prox_nuclei_count': result['prox_nuclei_count'],
                'total_nuclei_area': result['total_nuclei_area'],
                'average_distance': result['average_distance'],
                'prox_nuclei_ids': [nuc['nuclei_id'] for nuc in result['prox_nuclei']]
            })
        
        with open(output_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        
        print(f"Detailed results saved to {output_path}")
    
    def run_analysis(self, search_radius=100):
        print("Starting Pillar-Nuclei Spatial Analysis...")
        
        if not self.load_data():
            return
        print(f"\nAnalysing nuclei within {search_radius} pixels of wrapped pillars...")
        self.analyse_space(search_radius)
        
        self.summary()

        print("\nCreating plots...")
        self.create_plots()
        
        print("\nSaving detailed results...")
        self.save_detailed_results()
        
        print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    # Create analyser instance and run complete analysis
    analyser = PillarNucleiAnalyser()
    
    try:
        analyser.run_analysis(search_radius=100)
    except Exception as e:
        # Show error message if something goes wrong
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
        root.destroy()