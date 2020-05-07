print("[INFO] Starting...")
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import math
import cv2
import os

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# Loading our YOLO object detector trained on COCO dataset (80 classes),
# And determine only the *output* layer names that we need from YOLO.
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream, pointer to output video file, and
# frame dimensions
print("[INPUT] Do you want webcam (1) or test video (0)?")
n = int(input("Enter: "))
if n == 1:
    print("[INFO] Starting webcam...")
    cam = VideoStream(0).start()
else:
    path = input("Enter Path:")
    cam = VideoStream(src=path).start()
(W, H) = (None, None)

# loop over frames from the video file stream
i = 0
while True:
    i += 1
    if i % 2 == 0:
        continue

    # Reading frames
    (frame) = cam.read()
    # Loading Background Images
    image = cv2.imread("backimg.jpg")

    # Ensuring frame is not empty
    if frame is None:
        break

    # Resizing images
    frame = imutils.resize(frame, width=800)
    image = cv2.resize(image, (800, 450))

    # Initialize frame height and width
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Construct a blob from the input image and then perform a forward pass of
    # the YOLO object detector, giving us our bounding boxes and associated probabilities
    # [NOTE] For highly accurate detections use (256, 256) but frame rate will be less.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (128, 128),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    centroid = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Filter out other object expect person by ensuring
            # the class id is zero which indicates person
            if classID == 0 and confidence > 0.5:
                # Scale the bounding box coordinates back relative to
                # he size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates,
                centroid.append([centerX, centerY])
                boxes.append([x, y, int(width), int(height)])

    # Ensure at least one detection exists
    if len(boxes) > 0:
        # Initialize flag as zero which indicates no detection.
        flag = 0
        # Here we calculate distance between coordinates.
        # If distance is less the person height,
        # then display frame with warning, and turned flag equals to 1.
        for i, (x, y, w, h) in enumerate(boxes):
            for j, (centerX, centerY) in enumerate(centroid):
                if i == j:
                    continue
                # Calculating distance.
                dist = math.hypot((x + (w // 2)) - centerX, (y + (h // 2)) - centerY)

                # Checking distance is less then person's height or not.
                if h >= dist > 30:
                    flag = 1
                    # draw a point and line
                    cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), -1)
                    cv2.line(image, (centerX, centerY), ((x + (w // 2)), (y + (h // 2))), (0, 0, 255), (1))
                    break

            # draw a bounding box rectangle and point with different color
            # based on flag value
            if flag == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.circle(image, (x + (w // 2), y + (h // 2)), 5, (0, 0, 255), -1)
                flag = 0
                print("[WARNING] Violation of Social Distancing!!")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(image, (x + (w // 2), y + (h // 2)), 5, (0, 255, 0), -1)
    # Displaying output
    cv2.imshow("1", frame)
    cv2.imshow("2", image)
    c = cv2.waitKey(1)
    if c == 27:
        break

print("[INFO] Closing...!")
cv2.destroyAllWindows()
