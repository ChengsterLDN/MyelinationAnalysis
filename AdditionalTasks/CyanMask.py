import cv2
import numpy as np

img = cv2.imread('cell_38.png')           
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Cyan hue is roughly between ~80 and ~100 in OpenCV's 0-180 hue scale.
lower = np.array([80, 110, 110])   # [H, S, V] — tweak saturation/value thresholds as needed
upper = np.array([100, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

cv2.imwrite('cyan_mask.png', mask)       # binary mask saved
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()