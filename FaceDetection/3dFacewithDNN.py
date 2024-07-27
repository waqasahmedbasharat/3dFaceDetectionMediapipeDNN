import cv2
import mediapipe as mp
import time

# Load MediaPipe face mesh module for face mesh estimation
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
prevTime = 0

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the face mesh model
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # Read a frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to estimate face mesh
        results = face_mesh.process(image_rgb)

        # Draw face mesh on the image
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

        # Calculate and display FPS
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        # Display the image
        cv2.imshow('MediaPipe Face Mesh - Face Detection and Recognition', image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
