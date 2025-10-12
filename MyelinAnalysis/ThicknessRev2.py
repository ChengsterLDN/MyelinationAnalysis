import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

image = cv2.imread("box_4.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_cyan = np.array([50, 40, 40])
upper_cyan = np.array([100, 255, 255])
cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

# Denoise
kernel = np.ones((3, 3), np.uint8)
cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel, iterations=1)
contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = cyan_mask.shape
max_centre_distance = 25
image_center = np.array([w/2, h/2])

min_area = 50  

ring_mask = np.zeros_like(cyan_mask)

if contours:
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue  

        # Get contour centroid
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # If centroid is close enough to the image centre, keep
        if np.linalg.norm(np.array([cx, cy]) - image_center) <= max_centre_distance:
            cv2.drawContours(ring_mask, [c], -1, 255, thickness=cv2.FILLED)

else:
    print("No contours found.")

# Intensity-based mask
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
intensity_mask = np.zeros_like(gray)

# Apply intensity threshold 
ring_pixels = gray[ring_mask == 255]
if len(ring_pixels) > 0:
    intensity_threshold = np.mean(ring_pixels)
    # Alternative: use median or percentile for different thresholding
    # intensity_threshold = np.median(ring_pixels)
    # intensity_threshold = np.percentile(ring_pixels, 60)  # Keep top 40% brightest
    
    # normalized intensity
    intensity_mask[(gray >= intensity_threshold) & (ring_mask == 255)] = 255
    
    print(f"Intensity threshold: {intensity_threshold:.2f}")
else:
    intensity_mask = ring_mask.copy()
    print("No ring pixels found for intensity analysis")

# Use intensity_mask instead of ring_mask for further processing
enhanced_ring_mask = intensity_mask.copy()

# Denoise
denoising_kernel = np.ones((3, 3), np.uint8)
enhanced_ring_mask = cv2.morphologyEx(enhanced_ring_mask, cv2.MORPH_OPEN, denoising_kernel, iterations=1)
enhanced_ring_mask = cv2.morphologyEx(enhanced_ring_mask, cv2.MORPH_CLOSE, denoising_kernel, iterations=1)

# Remove small noise components
contours, _ = cv2.findContours(enhanced_ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 10 
for contour in contours:
    if cv2.contourArea(contour) < min_contour_area:
        cv2.drawContours(enhanced_ring_mask, [contour], -1, 0, thickness=cv2.FILLED)


skeleton = skeletonize(enhanced_ring_mask.astype(bool)).astype(np.uint8) * 255
labeled = label(skeleton > 0, connectivity=2)
props = regionprops(labeled)

if props:
    largest_label = max(props, key=lambda x: x.area).label
    clean_skeleton = (labeled == largest_label).astype(np.uint8) * 255
else:
    clean_skeleton = skeleton.copy()  

dist_transform = cv2.distanceTransform(enhanced_ring_mask, cv2.DIST_L2, 3)
thickness_values = dist_transform[clean_skeleton == 255] * 2 

skeleton_overlay = cv2.cvtColor(enhanced_ring_mask, cv2.COLOR_GRAY2BGR)
skeleton_overlay[clean_skeleton == 255] = [0, 0, 255]  

# Compute ring area 
ring_area_pixels = np.sum(enhanced_ring_mask > 0)
pixel_conv = 244.86 / 2048 
ring_area_um2 = ring_area_pixels * (pixel_conv**2)

if len(thickness_values) > 0:
    avg_thickness = pixel_conv * np.mean(thickness_values)
    std_thickness = pixel_conv * np.std(thickness_values)
    print(f"Average ring thickness: {avg_thickness:.2f} microns")
    print(f"Thickness std dev: {std_thickness:.2f} microns")
    print(f"Area: {ring_area_um2:.2f} microns squared")
else:
    avg_thickness = std_thickness = 0
    print("No thickness values found.")

# Create output - keep only intense pixels, make others black
output = image.copy()
output[enhanced_ring_mask == 0] = 0

cv2.namedWindow("Isolated Ring", cv2.WINDOW_NORMAL)
cv2.imshow("Isolated Ring", output)
cv2.namedWindow("Ring Mask", cv2.WINDOW_NORMAL)
cv2.imshow("Ring Mask", enhanced_ring_mask)
cv2.namedWindow("Skeleton", cv2.WINDOW_NORMAL)
cv2.imshow("Skeleton", clean_skeleton)
cv2.namedWindow("Skeleton Overlay", cv2.WINDOW_NORMAL)
cv2.imshow("Skeleton Overlay", skeleton_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()