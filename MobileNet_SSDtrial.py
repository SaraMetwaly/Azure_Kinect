import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Load class names (for MobileNet SSD)
with open("synset_words.txt", "r") as f:  # Replace with your path to class names file
    classes = [line.strip() for line in f.readlines()]

# Initialize the Azure Kinect SDK
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED))
k4a.start()

while True:
    # Get the next capture
    capture = k4a.get_capture()

    # Extract the color image and depth image
    color_image = capture.color
    depth_image = capture.depth

    if color_image is not None and isinstance(color_image, np.ndarray):
        # Prepare the image for MobileNet SSD
        blob = cv2.dnn.blobFromImage(color_image, scalefactor=1.0, size=(300, 300), 
                                     swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Process the detections
        height, width = color_image.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x2, y2) = box.astype("int")

                # Draw bounding box and label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(color_image, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Get the center point of the object
                center_x = (x + x2) // 2
                center_y = (y + y2) // 2

                # Ensure the center point is within the bounds of the depth image
                if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                    # Get the distance from the depth image at the center point
                    depth_value = depth_image[center_y, center_x]
                    distance_meters = depth_value / 1000.0
                    cv2.putText(color_image, f"Distance: {distance_meters:.2f} m", (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image with object detection and distance
    cv2.imshow("Kinect Color Image", color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and release resources
k4a.stop()
cv2.destroyAllWindows()
