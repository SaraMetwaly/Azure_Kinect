from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2

# Function to calculate angle between two lines formed by points
def calculate_angle(p1, p2, p3, p4):
    # Vector p1p2
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    # Vector p3p4
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
    # Calculate the angle between vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_theta = dot_product / (norm_v1 * norm_v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip value to avoid out of range errors
    
    return np.degrees(theta)  # Convert angle from radians to degrees

# Load camera with the default config
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
k4a.start()

# Dummy arm detection coordinates (replace with actual arm detection)
# Coordinates for two points on each arm
left_arm = [(100, 200), (150, 250)]
right_arm = [(100, 300), (250, 250)]

while True:
    # Get the next capture
    capture = k4a.get_capture()

    # Extract the color image and depth image
    color_image = capture.color
    depth_image = capture.depth

    # Check if the color and depth images are valid
    if color_image is not None and isinstance(color_image, np.ndarray):
        # Draw the detected arm points (replace with actual detected points)
        for point in left_arm + right_arm:
            cv2.circle(color_image, point, 5, (0, 255, 0), -1)
        
        # Draw lines between points on each arm
        cv2.line(color_image, left_arm[0], left_arm[1], (255, 0, 0), 2)
        cv2.line(color_image, right_arm[0], right_arm[1], (255, 0, 0), 2)
        
        # Calculate angle between left arm and right arm
        angle = calculate_angle(left_arm[0], left_arm[1], right_arm[0], right_arm[1])
        
        # Display the angle on the image
        cv2.putText(color_image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the image with arm detection and angle
    cv2.imshow("Kinect Color Image", color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and release resources
k4a.stop()
cv2.destroyAllWindows()
