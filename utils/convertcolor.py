import cv2

# Read image (OpenCV reads images as BGR by default)
image = cv2.imread('test0.jpg')
print(image.shape)
# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# # Convert RGB back to BGR (if needed)
# bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Save the result
cv2.imwrite('test00.jpg', rgb_image)