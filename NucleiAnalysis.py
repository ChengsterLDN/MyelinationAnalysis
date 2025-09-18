import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

folder_path = 'C:\\Users\\jonat\\Myelination\\LIFAccess\\PDL_extracted_pngs\\image_0\\nuclei'
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
# If your image is already grayscale, this is perfect.
#mip_gray = cv2.imread('maximum_intensity_projection.png', 0)

# Convert the image to grayscale
mip_gray = cv2.cvtColor(mip, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
# The function returns the optimal threshold value and the thresholded image
otsu_threshold, binary_image = cv2.threshold(mip_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Adjust these parameters based on your image
OPENING_KERNEL_SIZE = 3    # Larger = more aggressive speckle removal
CLOSING_KERNEL_SIZE = 3    # Larger = more aggressive hole filling
MIN_OBJECT_AREA = 50       # Minimum size for objects to keep

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

print(f"The automatically calculated Otsu threshold is: {otsu_threshold}")

# Save the binary image
cv2.imwrite('mip_binary_otsu.png', binary_image)

# Display the results (optional)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original MIP (Grayscale)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.hist(mip_gray.ravel(), 256, [0,256])
plt.axvline(x=otsu_threshold, color='r', linestyle='--', label=f'Otsu Threshold: {otsu_threshold}')
plt.title('Histogram & Threshold')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(final_binary, cmap='gray')
plt.title('Otsu Binary Image')
plt.axis('off')

plt.tight_layout()
plt.show()