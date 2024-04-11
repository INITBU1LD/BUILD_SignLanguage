import cv2
import av
import streamlit as st
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, VideoProcessorBase

from streamlit_pills import pills

# ------------------------------ Streamlit page configuration ------------------------------ #
st.set_page_config(
    page_title="SLR",
    page_icon=":the_horns:"
)
st.title("🤘 Sign Language Recognition")
st.subheader("Powered by :blue[Tensorflow] + :blue[Mediapipe] + :blue[Google Data]")

# Team Members
team = [
    "Diego Diaz",
    "Emily Salgueiros",
    "Gabriela Saldana",
    "Ian De Leon",
    "Ishaan Kalbhor",
    "Ivan Figueroa",
    "James Bustos",
    "Luis Canessa",
    "Nicolas Astros",
    "Nicole Sanchez",
    "Rebeca serralta",
    "Riguens Brutus",
    "Uriel Juarez",
    "mojeed ashaleye",
]

st.text("Team Members:")
category = pills("", team, None, index=None, label_visibility="collapsed", clearable=True)


# ------------------------------ Load Model ------------------------------ #

# Load TF Lite Model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

# Load MediaPipe Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Add Labels
train = pd.read_csv('train.csv')

# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# ------------------------------ Helper Functions ------------------------------ #

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

# ------------------------------ Streamlit WebRTC ------------------------------ #

class MPVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.frame_keypoints = []
        self.latest_prediction = ""
        self.confidence_threshold = 0.5  

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img, results = mediapipe_detection(img, self.model)
        draw_landmarks(img, results)
        keypoints = extract_keypoints(results)
        self.frame_keypoints.append(keypoints)
        self.frame_keypoints = self.frame_keypoints[-30:]

        if len(self.frame_keypoints) >= 30:
            res = np.expand_dims(self.frame_keypoints, axis=0)[0].astype(np.float32)
            self.frame_keypoints = []
            prediction = prediction_fn(inputs=res)
            probabilities = prediction['outputs'][0] 
            predicted_sign = np.argmax(probabilities)
            confidence = probabilities[predicted_sign]

            if confidence > self.confidence_threshold:
                confidence_pct = int(confidence * 100)  
                self.latest_prediction = f"{ORD2SIGN[predicted_sign]} ({confidence_pct}%)"
            else:
                self.latest_prediction = ""


        # Display the latest prediction on the video frame
        cv2.putText(img, f"Sign: {self.latest_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize the Streamlit WebRTC component.
webrtc_streamer(key="mpstream", video_processor_factory=MPVideoProcessor,
                             video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}, muted=True))

# ------------------------------ Streamlit Components ------------------------------ #

