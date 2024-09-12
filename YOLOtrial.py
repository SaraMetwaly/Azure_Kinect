from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with your paths to YOLO weights and config
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:  # Replace with your path to class names file
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
        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(color_image, scalefactor=0.00392, size=(416, 416), 
                                     swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(obj[0] * color_image.shape[1])
                        center_y = int(obj[1] * color_image.shape[0])
                        w = int(obj[2] * color_image.shape[1])
                        h = int(obj[3] * color_image.shape[0])

                        # Get bounding box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Get the center point of the object
            center_x = x + w // 2
            center_y = y + h // 2

            # Ensure the center point is within the bounds of the depth image
            if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                # Get the distance from the depth image at the center point
                depth_value = depth_image[center_y, center_x]
                distance_meters = depth_value / 1000.0
                cv2.putText(color_image, f"Distance: {distance_meters:.2f} m", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the image with object detection and distance
    cv2.imshow("Kinect Color Image", color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and release resources
k4a.stop()
cv2.destroyAllWindows()
