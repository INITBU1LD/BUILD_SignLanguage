import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converts the image from BGR to RGB color space for MediaPipe processing.
    image.flags.writeable = False  # Optimizes performance by making the image non-writeable.
    results = model.process(image)  # Processes the image to detect landmarks.
    image.flags.writeable = True  # Makes the image writeable again.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converts the image back to BGR color space for further processing or display.
    return image, results
cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read() #Read Feed
        image, results = mediapipe_detection(frame, holistic) #Make detections
        print(results)
        draw_styled_landmarks(image, results) #Draw landmarks
        cv2.imshow('OpenCV Feed', image) #Show to screen
        if cv2.waitKey(10) & 0xFF == ord('q'): #Break gracefully
            break
    cap.release()
    cv2.destroyAllWindows()

