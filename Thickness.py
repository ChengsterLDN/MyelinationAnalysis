import cv2
import numpy as np
from skimage.morphology import skeletonize

# Load the combined image
image = cv2.imread("cell_29.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Isolate yellow
lower_yellow = np.array([20, 100, 100])  
upper_yellow = np.array([30, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

not_yellow = cv2.bitwise_not(yellow_mask)

# Isolate ring
lower_cyan = np.array([85, 100, 100])  
upper_cyan = np.array([95, 255, 255])
lower_green = np.array([40, 100, 100])  
upper_green = np.array([70, 255, 255])

cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
green_mask = cv2.inRange(hsv, lower_green, upper_green)
ring_mask = cv2.bitwise_or(cyan_mask, green_mask)


kernel = np.ones((3,3), np.uint8)
ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Skeletonize and measure thickness
skeleton = skeletonize(ring_mask.astype(bool)).astype(np.uint8) * 255
dist_transform = cv2.distanceTransform(ring_mask, cv2.DIST_L2, 3)
thickness_values = dist_transform[skeleton == 255] * 2  


avg_thickness = np.mean(thickness_values)
std_thickness = np.std(thickness_values)
print(f"Average ring thickness: {avg_thickness:.2f} pixels")
print(f"Thickness std dev: {std_thickness:.2f} pixels")


output = image.copy()
output[ring_mask == 0] = 0  
cv2.putText(output, f"Avg Thickness: {avg_thickness:.2f}px", (10,30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

cv2.imshow("Isolated Ring", output)
cv2.imshow("Thickness Mask", ring_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()