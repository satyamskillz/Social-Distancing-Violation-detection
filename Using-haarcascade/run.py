# Import required library
print("[INFO] Starting...")
from imutils.video import VideoStream
import imutils
import cv2
import math


# Defined the function to calculate centroid of box.
def centre(body):
    x = body[0]
    y = body[1]
    w = body[2]
    h = body[3]
    return x + (w // 2), y + (h // 2)


# Start webcam or video
print("[INPUT] Do you want webcam (1) or test video (0)?")
n = int(input("Enter: "))
if n == 1:
    print("[INFO] Starting webcam...")
    cam = VideoStream(0).start()
else:
    path = input("Enter path: ")
    cam = VideoStream(src=path).start()

# Loading pre-trained haarcascade_upperbody model for upper body detection
# Note - haarcascade_fullbody model can be also used but it is failing to detect to person on bike and bicycle
print("[INFO] Loading Model...")
bodyModel = cv2.CascadeClassifier("models/haarcascade_upperbody.xml")

# loop over frames from the video file stream
while True:
    # Reading frames
    frames = cam.read()

    # Initialize empty dictionary to store point for each detected person
    head_points = {}

    # loading background image
    image = cv2.imread("backimg.jpg")

    # Resizing images
    image = imutils.resize(image, width=500)
    frame = imutils.resize(frames, width=500)

    # Converting RGB image into grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect person in grayscale image
    # Output is 2D list, row indicates number of person and column indicates coordinates of person [box].
    persons = bodyModel.detectMultiScale(gray, 1.03, 4)

    # Ensure at least one person in the frame.
    if len(persons) > 0:
        for i, body in enumerate(persons):
            # Store coordinates of person in variables.
            x = body[0]
            y = body[1]
            w = body[2]
            h = body[3]

            # Initialize flag as zero which indicates no detection.
            flag = 0

            # Calculate centroid of box.
            (midx, midy) = centre(body)

            # Ensure at least one person box is stored in dictionary.
            # Here we calculate distance between coordinates.
            # If distance is less the 60 pixels,
            # then display frame with warning, and turned flag equals to 1.
            if len(head_points) > 0:
                for j, (xc, yc) in enumerate(head_points.values()):

                    # Calculating distance.
                    dist = math.hypot(xc - midx, yc - midy)

                    # Check distance is less then 60 pixels or not.
                    if dist <= 60:

                        # Draw points on images.
                        cv2.circle(image, ((xc + midx) // 2, (yc + midy) // 2), 30, (255, 0, 0), -1)
                        cv2.circle(image, (xc, yc), 5, (0, 0, 255), -1)
                        cv2.circle(image, (midx, midy), 5, (0, 0, 255), -1)
                        flag = 1
                        print("[WARNING] Violation of distance!")
            # Store centroid into dictionary.
            head_points[i] = (midx, midy)

            # Draw red rectangle if flag is one,
            # else draw green rectangle and points
            if flag == 0:
                cv2.circle(image, (midx, midy), 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display images
    cv2.imshow("video", frame)
    cv2.imshow("image", image)
    c = cv2.waitKey(1)
    if c == 27:
        break
print("[INFO] Closing....!")
cv2.destroyAllWindows()
