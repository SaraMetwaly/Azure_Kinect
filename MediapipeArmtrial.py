# import cv2
# import numpy as np
# from pyk4a import PyK4A, Config, ColorResolution, DepthMode
# from matplotlib import pyplot as plt

# # Initialize Azure Kinect camera
# k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
# k4a.start()

# def calculate_angle(p1, p2, p3, p4):
#     # Vector p1p2
#     v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
#     # Vector p3p4
#     v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
#     # Calculate the angle between vectors
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
    
#     cos_theta = dot_product / (norm_v1 * norm_v2)
#     theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip value to avoid out of range errors
    
#     return np.degrees(theta)  # Convert angle from radians to degrees

# while True:
#     # Get the next capture
#     capture = k4a.get_capture()

#     # Extract the color image
#     color_image = capture.color
#     if color_image is None:
#         continue

#     # Convert the color image to RGB
#     image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

#     # Process the image with your arm detection algorithm
#     # Replace this part with your custom arm detection code
#     # For demonstration, let's assume you have the following dummy points
#     arm_points = [
#         [(100, 100), (200, 200)],  # Coordinates for arm 1
#         [(300, 300), (400, 400)]   # Coordinates for arm 2
#     ]

#     # Calculate the angle between the arms
#     if len(arm_points) == 2:
#         angle = calculate_angle(
#             arm_points[0][0], arm_points[0][1],  # Arm 1 points
#             arm_points[1][0], arm_points[1][1]   # Arm 2 points
#         )
        
#         # Display the angle on the image
#         cv2.putText(color_image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the image
#     cv2.imshow('Azure Kinect Color Image', color_image)

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Stop the camera and release resources
# k4a.stop()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode

# Initialize Azure Kinect camera
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
k4a.start()

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

def detect_and_track_arms(image):
    # Placeholder for custom arm detection logic
    # Replace this with your specific arm detection algorithm
    # Example format: [(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]
    # Each tuple represents a pair of points on detected arms
    return [((100, 100), (200, 200)), ((300, 300), (400, 400))]

while True:
    # Get the next capture
    capture = k4a.get_capture()

    # Extract the color image
    color_image = capture.color
    if color_image is None:
        continue

    # Convert the color image to RGB
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Detect and track arms
    arm_points = detect_and_track_arms(image_rgb)
    
    if len(arm_points) == 2:
        # Calculate the angle between the lines formed by the arm points
        angle = calculate_angle(
            arm_points[0][0], arm_points[0][1],  # Arm 1 points
            arm_points[1][0], arm_points[1][1]   # Arm 2 points
        )
        
        # Display the angle on the image
        cv2.putText(color_image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Azure Kinect Color Image', color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and release resources
k4a.stop()
cv2.destroyAllWindows()
