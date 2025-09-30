import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

folder_path = 'C:\\Users\\jonat\\Documents\\My Documents\\MecBioMed\\MyelinationProject\\MBP DATA\\MBP V5 coating\\PDL\\PDL_extracted_pngs\\image_0\\nuclei'

#folder_path = 'C:\\Users\\Jonathon Cheng\\OneDrive - University College London\\Documents\\UCL\\Summer 2025\\MyelinationProject\\MBP V5 coating\\PDL\\PDL_extracted_pngs\\image_0\\nuclei'
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

# Initialize an empty list to hold the images as arrays
image_stack = []

for filename in file_list:
    filepath = os.path.join(folder_path, filename)
    # Read the image. cv2.imread reads in BGR color order by default.
    img = cv2.imread(filepath)
    image_stack.append(img)

# Convert the list to a NumPy array
image_stack = np.array(image_stack)

# Perform the max projection
mip = np.max(image_stack, axis=0)

# OpenCV uses BGR order, so if you want RGB for saving or display, convert it.
# If your PNGs are grayscale, skip this step.
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

# Adjust 
OPENING_KERNEL_SIZE = 2 # Larger = more aggressive speckle removal
CLOSING_KERNEL_SIZE = 2    # Larger = more aggressive hole filling
MIN_OBJECT_AREA = 10       # Minimum size for objects to keep

# Create kernels
opening_kernel = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
closing_kernel = np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), np.uint8)


binary_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, closing_kernel)
binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, opening_kernel)


# Step 3: Remove very small objects by area
contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_mask = np.zeros_like(binary_cleaned)

for cnt in contours:
    if cv2.contourArea(cnt) > MIN_OBJECT_AREA:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

final_binary = filtered_mask


# Count the number of particles in the final binary mask
contours_final, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define size thresholds (adjust better)
MIN_SIZE_THRESHOLD = 700   # Minimum area in pixels for "large" particles
MAX_SIZE_THRESHOLD = 5000   # Max area

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
    print(f"Number of particles analyzed: {len(circularities)}")
    
    # Count particles by circularity ranges
    high_circularity = sum(1 for c in circularities if c >= 0.8)
    medium_circularity = sum(1 for c in circularities if 0.6 <= c < 0.8)
    low_circularity = sum(1 for c in circularities if c < 0.6)
    
    print(f"\nCircularity Categories:")
    print(f"High (≥0.8): {high_circularity} particles ({high_circularity/len(circularities)*100:.1f}%)")
    print(f"Medium (0.6-0.8): {medium_circularity} particles ({medium_circularity/len(circularities)*100:.1f}%)")
    print(f"Low (<0.6): {low_circularity} particles ({low_circularity/len(circularities)*100:.1f}%)")
else:
    print("No valid particles for circularity analysis")

    
# Create visualisation with large particles coloured differently
size_filtered_image = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours_final):
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Colour code based on size: red for large, yellow for small
        if area >= MIN_SIZE_THRESHOLD and area <= MAX_SIZE_THRESHOLD:
            color = (0, 0, 255)  # Red for particles within size range
            cv2.putText(size_filtered_image, f"{i+1}({int(area)})", (cx-15, cy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        else:
            color = (0, 255, 255)  # Yellow for particles outside size range
            cv2.putText(size_filtered_image, f"{i+1}", (cx-5, cy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        cv2.drawContours(size_filtered_image, [cnt], -1, color, 1)

# Save the size-filtered results
cv2.imwrite('large_particles_only.png', large_particles_image)
cv2.imwrite('size_filtered_particles.png', size_filtered_image)

print(f"The automatically calculated Otsu threshold is: {otsu_threshold}")

# Save the binary image
cv2.imwrite('mip_binary_otsu.png', binary_image)


# Display 
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Otsu Binary')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(large_particles_image, cmap='gray')
plt.title(f'Cleaned Binary\n{num_large_particles} particles detected')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(size_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Particles Numbered')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.hist(mip_gray.ravel(), 256, [0,256])
plt.axvline(x=otsu_threshold, color='r', linestyle='--', label=f'Otsu Threshold: {otsu_threshold}')
plt.title('Histogram & Threshold')
plt.legend()

plt.tight_layout()
plt.show()
