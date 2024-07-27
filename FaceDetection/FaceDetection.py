import cv2
import mediapipe as mp
import time

# Load Mediapipe face detection module
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
prevTime = 0

# Initialize the webcam
# cap = cv2.VideoCapture(0)

# For Video input:
cap = cv2.VideoCapture("1.mp4")

# Initialize the face detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        # Read a frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect faces
        results = face_detection.process(image_rgb)

        # Draw face detections and labels on the image
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (255, 0, 255), 2)
                cv2.putText(image, 'Face Detected', (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Calculate and display FPS
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        # Display the image
        cv2.imshow('BlazeFace Face Detection', image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
