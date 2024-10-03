import numpy as np
import cv2
from pyk4a import PyK4A, Config, ColorResolution, DepthMode

# Initialize Kinect camera with correct enums for resolution and depth mode
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
k4a.start()

# Capture a frame from the camera
capture = k4a.get_capture()

# Retrieve the color and depth images
color_image = capture.color  # This is the RGB image
depth_image = capture.depth  # This is the depth map

# Convert the color image to HSV (useful for color-based object detection)
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# Thresholding to detect objects of a specific color (example: detecting red objects)
# Adjust the color ranges to match the object you want to detect
lower_bound = np.array([0, 100, 100])  # Lower bound of red in HSV
upper_bound = np.array([10, 255, 255])  # Upper bound of red in HSV
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Find contours (detected objects) from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Function to extract 3D coordinates (x, y, z) from the depth image
def get_3d_coordinates(depth_image, x, y):
    z = depth_image[y, x]  # Depth value (z-axis)
    return np.array([x, y, z])

# Check if we detected at least two objects (two largest contours)
if len(contours) >= 2:
    # Sort contours by area and take the two largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Get the centroid of the two largest objects
    def get_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        else:
            return None

    # Get centroids of two largest objects
    object1_centroid = get_centroid(contours[0])
    object2_centroid = get_centroid(contours[1])

    if object1_centroid and object2_centroid:
        # Extract 3D coordinates of the centroids using the depth image
        object1_coords = get_3d_coordinates(depth_image, *object1_centroid)
        object2_coords = get_3d_coordinates(depth_image, *object2_centroid)

        # Compute the vector between the two objects
        vector1 = object1_coords
        vector2 = object2_coords

        # Compute the angle between the vectors
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate the angle in radians, then convert to degrees
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
        angle_degrees = np.degrees(angle_radians)

        print(f"Angle between objects: {angle_degrees:.2f} degrees")
    else:
        print("Unable to find centroids for both objects.")
else:
    print("Less than two objects detected.")

# Display the results (for debugging purposes)
cv2.imshow("Original Image", color_image)
cv2.imshow("Object Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Stop the Kinect camera
k4a.stop()
