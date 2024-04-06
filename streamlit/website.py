import streamlit as st
import mediapipe as mp
import cv2
from PIL import Image
import numpy as np
import tempfile
import tensorflow as tf
import os
import tflite_runtime.interpreter as tflite
#from tensorflow.keras.layers import TFSavedModelLayer 


#Load TF Lite Model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")



LSTM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.tflite')

interpreter = tflite.Interpreter(LSTM_PATH)
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")
# Get the inference function from the loaded model
# infer = interpreter.signatures["serving_default"]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, model_holistic, model_tf):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model_holistic.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Drawing landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    # Additional landmarks drawing...
    # TensorFlow model prediction logic (adapt as necessary)
    # pred = model_tf.predict(preprocess_your_frame_here(frame))
    
    return frame

# Streamlit UI setup
st.title("Sign Language Recognition Project")
st.write("""This project aims to translate sign language gestures into text using deep learning.
A real-time solution for improving communication.""")

# Setup for webcam capture and image display
col1, col2 = st.columns([3, 1])
with col2:
    # sign_language_image = Image.open('C:\\Users\\rebec\\Pictures\\OIP.jpg')
    # st.image(sign_language_image, caption="Try to copy these signs")
    pass

# Start/Stop Stream logic
if st.button('Start Stream', key='start'):
    cap = cv2.VideoCapture(0)
    tfile = tempfile.NamedTemporaryFile(delete=False)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, holistic, loaded_model)
            
            cv2.imwrite(tfile.name, processed_frame)
            col1.image(tfile.name)

            if st.session_state.get('stop_pressed', False):
                st.session_state['stop_pressed'] = False
                break

    cap.release()
    tfile.close()

if st.button('Stop Stream', key='stop'):
    st.session_state['stop_pressed'] = True
