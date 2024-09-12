from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import numpy as np  # Import numpy
from matplotlib import pyplot as plt

# Load camera with the default config
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
k4a.start()

# Load the face detection model from OpenCV (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Get the next capture
    capture = k4a.get_capture()

    # Extract the color image and depth image
    color_image = capture.color
    depth_image = capture.depth

    # Check if the color and depth images are valid
    if color_image is not None and isinstance(color_image, np.ndarray):
        # Convert color image to grayscale for face detection
        gray_image = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_BGR2GRAY)  # Drop the alpha channel if present

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get the center point of the face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Ensure the center point is within the bounds of the depth image
            if 0 <= face_center_x < depth_image.shape[1] and 0 <= face_center_y < depth_image.shape[0]:
                # Get the distance from the depth image at the center point of the face
                depth_value = depth_image[face_center_y, face_center_x]

                # Convert depth value to meters (assuming Kinect depth is in millimeters)
                distance_meters = depth_value / 1000.0

                # Display the distance on the image
                cv2.putText(color_image, f"Distance: {distance_meters:.2f} m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the image with face detection and distance
    cv2.imshow("Kinect Color Image", color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and release resources
k4a.stop()
cv2.destroyAllWindows()
