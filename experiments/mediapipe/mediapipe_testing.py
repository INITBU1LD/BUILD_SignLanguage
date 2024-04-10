# Necessary Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from Helper import extract_keypoints

# MediaPipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to detect landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converts the image from BGR to RGB color space for MediaPipe processing.
    image.flags.writeable = False  # Optimizes performance by making the image non-writeable.
    results = model.process(image)  # Processes the image to detect landmarks.
    image.flags.writeable = True  # Makes the image writeable again.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converts the image back to BGR color space for further processing or display.
    return image, results

# Function to draw landmarks
def draw_styled_landmarks(image, results):
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
cap = cv2.VideoCapture(0)

# OpenCV feed and mediapipe detection
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read() #Read Feed
        image, results = mediapipe_detection(frame, holistic) #Make detections
        print(results)
        draw_styled_landmarks(image, results) #Draw landmarks
        cv2.imshow('OpenCV Feed', image) #Show to screen
        data = extract_keypoints(results)
        if cv2.waitKey(10) & 0xFF == ord('q'): #Break gracefully
            break
    cap.release()
    cv2.destroyAllWindows()