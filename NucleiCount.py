import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#folder_path = 'C:\\Users\\jonat\\OneDrive - University College London\\Documents\\UCL\\Summer 2025\\MyelinationProject\\PDL_extracted_pngs\\image_8\\nuclei'

#folder_path = 'C:\\Users\\Jonathon Cheng\\OneDrive - University College London\\Documents\\UCL\\Summer 2025\\MyelinationProject\\PDL_extracted_pngs\\image_1\\nuclei'


mip_path = 'C:\\Users\\jonat\\OneDrive - University College London\\Documents\\UCL\\Summer 2025\\MyelinationProject\\240925\\DMSO-5\\MAX_3D pup OL 3doses  test t3 24sept25.lif - DMSO-5 - C=0.png'  # Add your MIP image path here
mip = cv2.imread(mip_path)

if mip is None:
    print(f"Error: Could not load MIP image from {mip_path}")
    exit()

# OpenCV uses BGR order, so if you want RGB for saving or display, convert it.
# If your PNGs are grayscale, skip 
mip_rgb = cv2.cvtColor(mip, cv2.COLOR_BGR2RGB)

# Save the image using OpenCV
#cv2.imwrite('mip_opencv.png', mip) # saves in BGR order
# Or, to save in RGB, use the converted image and ensure it's uint8
# cv2.imwrite('mip_opencv_rgb.png', cv2.cvtColor(mip, cv2.COLOR_RGB2BGR))

# Read the MIP image in grayscale mode (0 flag)
#mip_gray = cv2.imread('maximum_intensity_projection.png', 0)

# Convert the image to grayscale
mip_gray = cv2.cvtColor(mip, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
otsu_threshold, binary_image = cv2.threshold(mip_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


MIN_OBJECT_AREA = 20       # Minimum size for objects to keep

binary_cleaned = cv2.fastNlMeansDenoising(binary_image)


# Remove very small objects by area
contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_binary = np.zeros_like(binary_cleaned)

for cnt in contours:
    if cv2.contourArea(cnt) > MIN_OBJECT_AREA:
        cv2.drawContours(final_binary, [cnt], -1, 255, -1)

# Count the number of particles in the final binary mask
contours_final, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define size thresholds (adjust better)
MIN_SIZE_THRESHOLD = 450   # Minimum area in pixels for "large" particles
MAX_SIZE_THRESHOLD = 7500   # Max area

# Count particles above size threshold
large_particles = []
particle_areas = []

for cnt in contours_final:
    area = cv2.contourArea(cnt)
    particle_areas.append(area)
    if area >= MIN_SIZE_THRESHOLD and area <= MAX_SIZE_THRESHOLD:
        large_particles.append(cnt)

num_large_particles = len(large_particles)
print(f"Number of particles: {num_large_particles}")

# Create separate images for large particles only
large_particles_image = np.zeros_like(final_binary)
cv2.drawContours(large_particles_image, large_particles, -1, 255, -1)

# Calculate circularity for each particle
circularities = []
valid_particles = []  

for i, cnt in enumerate(contours_final):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Avoid division by zero and invalid contours
    if perimeter > 0 and area > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularities.append(circularity)
        valid_particles.append(cnt)
    else:
        print(f"Particle {i+1}: Invalid contour (area: {area}, perimeter: {perimeter})")

# Calculate statistics
if circularities:
    avg_circularity = np.mean(circularities)
    median_circularity = np.median(circularities)
    min_circularity = np.min(circularities)
    max_circularity = np.max(circularities)
    std_circularity = np.std(circularities)
    
    print("\n=== CIRCULARITY ANALYSIS ===")
    print(f"Average circularity: {avg_circularity:.3f}")
    print(f"Median circularity: {median_circularity:.3f}")
    print(f"Minimum circularity: {min_circularity:.3f}")
    print(f"Maximum circularity: {max_circularity:.3f}")
    print(f"Standard deviation: {std_circularity:.3f}")
    print(f"Number of particles analysed: {len(circularities)}")
    
    # Count particles by circularity ranges
    high_circularity = sum(1 for c in circularities if c >= 0.8)
    low_circularity = sum(1 for c in circularities if 0.6 <= c < 0.8)
    anomalous_circularity = sum(1 for c in circularities if c < 0.01)
    num_large_particles = num_large_particles - anomalous_circularity
    print(num_large_particles)
else:
    print("No valid particles for circularity analysis")

    
# Create visualisation with large particles coloured differently
size_filtered_image = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours_final):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Colour code based on size: red for large, yellow for small
        if area >= MIN_SIZE_THRESHOLD and area <= MAX_SIZE_THRESHOLD and circularity >= 0.01:
            color = (0, 0, 255)  # Red for particles within size range
            cv2.putText(size_filtered_image, f"{i+1}({int(area)})", (cx-15, cy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        else:
            color = (0, 255, 255)  # Yellow for particles outside size and circularity range
            cv2.putText(size_filtered_image, f"{i+1}", (cx-5, cy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        cv2.drawContours(size_filtered_image, [cnt], -1, color, 1)

# Save 

cv2.imwrite('NucleiCount.png', size_filtered_image)



# Display 
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Otsu Binary')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(size_filtered_image, cv2.COLOR_BGR2RGB))
plt.title(f'Cleaned Binary\n{num_large_particles} particles detected')
plt.axis('off')

"""plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(size_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Particles Numbered')
plt.axis('off')"""

"""plt.subplot(1, 4, 4)
plt.hist(mip_gray.ravel(), 256, [0,256])
plt.axvline(x=otsu_threshold, color='r', linestyle='--', label=f'Otsu Threshold: {otsu_threshold}')
plt.title('Histogram & Threshold')
plt.legend()"""

plt.tight_layout()
plt.show()
