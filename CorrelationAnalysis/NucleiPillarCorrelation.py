# yellow_centers.py
import cv2
import numpy as np
import os

image_path = 'test_pillar.png'

def detect_centres(image_path, output_file=None):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centres = []
    min_area = 100  # Minimum area to consider as valid geometry
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Calculate moments to find centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centre_x = int(M["m10"] / M["m00"])
                centre_y = int(M["m01"] / M["m00"])
                centres.append((centre_x, centre_y))
    centres.sort(key=lambda point: point[0])
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.splitext(image_path)[0]
        output_file = f"{base_name}_coordinates.csv"
    
    # Save coordinates to file
    with open(output_file, 'w') as f:
        f.write("Centres (x, y coordinates):\n")
        f.write("Format: x, y\n")
        f.write("=" * 40 + "\n")
        for i, (x, y) in enumerate(centres, 1):
            f.write(f"{i:3d}: {x:4d}, {y:4d}\n")
    
    print(f"Coordinates saved to: {output_file}")
    create_visualisation(image, centres, image_path)
    
    return centres

def create_visualisation(image, centres, original_path):
    vis_image = image.copy()
    
    # Mark centres red
    for i, (x, y) in enumerate(centres):
        cv2.circle(vis_image, (x, y), 10, (0, 0, 255), 2) 
        cv2.putText(vis_image, str(i+1), (x+15, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    base_name = os.path.splitext(original_path)[0]
    vis_path = f"{base_name}_detected.jpg"
    cv2.imwrite(vis_path, vis_image)
    print(f"Visualisation saved to: {vis_path}")

if __name__ == "__main__":
    print(f"Processing image: {image_path}")
    detect_centres(image_path)
    print("Centre computation complete!")