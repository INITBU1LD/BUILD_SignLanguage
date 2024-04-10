import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd

# ------------------ Load Models ------------------ #

# Load TF Lite Model
interpreter = tf.lite.Interpreter(model_path="./weights/model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

# Load MediaPipe Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Add Labels
train = pd.read_csv('asl-signs/train.csv')

# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# ------------------ Helper Functions ------------------ #

# Function to process the video frame with MediaPipe.
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Color conversion BGR 2 RGB
    frame.flags.writeable = False                  # Frame not writeable to improve performance
    results = model.process(frame)                 # Make prediction
    frame.flags.writeable = True                   # Frame writeable
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Color conversion RGB 2 BGR
    return frame, results

# Function to draw landmarks on the video frame.
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

# Function to extract landmarks from the MediaPipe results.
def extract_keypoints(results):
    # Extract landmarks and flatten into a single array, if landmarks are detected otherwise fill with NaN

    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.full(33*3, np.nan)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.full(468*3, np.nan)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full(21*3, np.nan)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full(21*3, np.nan)

    # Concatenate all the keypoints into a single flattened array
    all_keypoints = np.concatenate([face, lh, pose, rh])

    # Reshape the array
    reshaped_keypoints = np.reshape(all_keypoints, (543, 3))

    return reshaped_keypoints


# Initialize variables for sentence building and frame count
frame_keypoints = []
sentence = []
frame_count = 0
word = ''
last_word = None  # Track the last word added to the sentence

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        frame_keypoints.append(keypoints)
        frame_keypoints = frame_keypoints[-30:]  # Keep only the last 30 frames

        # Make prediction every 30 frames
        if len(frame_keypoints) == 30:
            res = np.expand_dims(frame_keypoints, axis=0)[0].astype(np.float32)
            prediction = prediction_fn(inputs=res)
            predicted_sign = prediction['outputs'].argmax()
            word = ORD2SIGN[predicted_sign]  # Current predicted word

        # Display the current word
        cv2.putText(image, word, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Increment frame count
        frame_count += 1

        # Add word to sentence after a certain number of frames, only if it's different from the last word
        if frame_count == 90: 
            if word != last_word:  # Check if the current word is different from the last added word
                sentence.append(word)
                last_word = word  # Update the last word
            frame_count = 0  # Reset frame count

        # Join the sentence list to a string and display it
        sentence_text = ' '.join(sentence)
        cv2.putText(image, sentence_text, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

