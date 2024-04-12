import time
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    layout="wide",  # makes the layout wide
)


# ------------------ Load Models ------------------ #
# Load TF Lite Model
interpreter = tf.lite.Interpreter(model_path="weights/model.tflite")
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
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.full(33*3, np.nan)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.full(468*3, np.nan)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full(21*3, np.nan)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full(21*3, np.nan)
    all_keypoints = np.concatenate([face, lh, pose, rh])
    reshaped_keypoints = np.reshape(all_keypoints, (543, 3))
    return reshaped_keypoints

# Initialize variables for sentence building and frame count
frame_keypoints = []
sentence = []
frame_count = 0
word = ''
last_word = None  # Track the last word added to the sentence
current_word_placeholder = "Current predicted value:"
current_word_text = st.empty()

current_sentence_placeholder = "Current sentence:"
current_sentence_text = st.empty()
st.title('Sign Language Recognition Project')
st.write(
    """This project aims to translate sign language gestures into text using deep learning.
A real-time solution for improving communication.\n""")

# Initialize recording state
is_recording = False

# Function to toggle recording state
def toggle_recording():
    global is_recording
    is_recording = not is_recording

# Function to stop recording
def stop_recording():
    global is_recording
    if is_recording:
        toggle_recording()

# Display start/stop recording button
start_button_placeholder = st.empty()
if not is_recording:
    if start_button_placeholder.button("Start Recording"):
        toggle_recording()
        start_button_placeholder.empty()  # Remove the start button when clicked
        stop_button = st.button("Stop Recording")
else:
    stop_button = st.button("Stop Recording")
    if stop_button:
        stop_recording()
        start_button_placeholder.button("Start Recording")  # Display the start button when stop button is clicked

# Display recording status
st.write("Recording status:", "Recording" if is_recording else "Not Recording")

cap = cv2.VideoCapture(0)
st_image_placeholder = st.empty()

st.write("Here are some words you can try:")

col1, col2, col3 = st.columns(3)

with col1:
    image_url = "https://www.lifeprint.com/asl101/gifs-animated/shhh.gif"
    width, height = 450, 475  # desired width and height

    html_img = f'<img src="{image_url}" width="{width}" height="{height}">'
    st.markdown(html_img, unsafe_allow_html=True)
    st.caption("Shhhh")

with col2:
    st.markdown("![Alt Text](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdGowNW4zMWE1djZoZm8yNTc0czV0OG45ZndmaWpkeGk2YmdwcXVyMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Jx3bJbEpuZFz3iBWEp/giphy.gif)")
    st.caption("Water")

with col3:
    st.markdown("![Alt Text](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeW85ZmVrMXNzY2Jodm5tOGR1cTExeGJuZHRkdjB4Zmlzd2RqbHBjNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MBd09LHGjbdgw4CTtn/giphy.gif)")
    st.caption("Milk")

st.info(
    """Feeling to use more signs? You can select and learn from the list below!\n""")
word_to_image = {

    "after" : "https://lifeprint.com/asl101/gifs/a/after-over-across.gif",
    "airplane" : "https://www.lifeprint.com/asl101/gifs/a/airplane-flying.gif ",
    "all" : "https://lifeprint.com/asl101/gifs/a/all-whole.gif ",
    "alligator" : "https://lifeprint.com/asl101/gifs/a/alligator.gif ",
    "animal" : "https://www.lifeprint.com/asl101/gifs-animated/animal.gif ",
    "any" : "https://lifeprint.com/asl101/gifs/a/any.gif ",
    "apple" : "https://media.giphy.com/media/l0HlHb4dtZZiMYEA8/giphy.gif ",
    "aunt":"https://th.bing.com/th/id/OIP.Yz5UUZdNTrVWXf72we_N6wHaHa?rs=1&pid=ImgDetMain ",
    "bad" : "https://lifeprint.com/asl101/gifs/b/bad.gif ",
    "balloon" : "https://media.giphy.com/media/26FL9yfajyobRXJde/giphy.gif ",
    "bath" : "https://media.giphy.com/media/l0MYPjjoeJbZVPmNO/giphy.gif ",
    "bed" : "https://lifeprint.com/asl101/gifs/b/bed-1.gif ",
    "bedroom" : "https://lifeprint.com/asl101/gifs/b/bedroom.gif ",
    "bee" : "https://lifeprint.com/asl101/gifs/b/bee.gif ",
    "better" : "https://lifeprint.com/asl101/gifs/b/better.gif ",
    "bird" : "https://lifeprint.com/asl101/gifs/b/bird.gif ",
    "blue" : "https://lifeprint.com/asl101/gifs/b/blue-1.gif ",
    "boat" : "https://lifeprint.com/asl101/gifs/b/boat-ship.gif ",
    "book" : "https://media.giphy.com/media/l0MYL43dl4pQEn3uE/giphy.gif ",
    "boy" : "https://lifeprint.com/asl101/gifs/b/boy.gif ",
    "brother" : "https://lifeprint.com/asl101/gifs/b/brother.gif ",
    "brown" : "https://lifeprint.com/asl101/gifs/b/brown.gif ",
    "bug" : "https://lifeprint.com/asl101/gifs/b/bug.gif ",
    "bye" : "https://c.tenor.com/vME77PObDN8AAAAC/asl-bye-asl-goodbye.gif ",
    "Call on phone" : "https://www.lifeprint.com/asl101/gifs/c/call-hearing.gif ",
    "car":"https://th.bing.com/th/id/OIP.wxw32OaIdqFt8f_ucHVoRgHaEH?w=308&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7 ",
    "carrot" : "https://media.giphy.com/media/l0HlDdvqxs1jsRtiU/giphy.gif ",
    "cat" : "https://lifeprint.com/asl101/gifs-animated/cat-02.gif ",
    "child" : "https://lifeprint.com/asl101/gifs/c/child.gif ",
    "chocolate":"https://www.bing.com/th/id/OGC.97e3055b2bb3ae04390df712fcd3344f?pid=1.7&rurl=https%3a%2f%2fi.pinimg.com%2foriginals%2f9f%2fa2%2fb5%2f9fa2b5064a72b5e46202d20848f1bf21.gif&ehk=izvOlFp25%2fx5NVTCmqVz0UOnZNOWy%2fAJJtzAhkZ8nTg%3d ",
    "clean" : "https://media.giphy.com/media/3o7TKoturrdpf5Muwo/giphy.gif  ",
    "close":"https://www.bing.com/th/id/OGC.7fb3c4e8d1e6870e46b2a65ce8b0284f?pid=1.7&rurl=https%3a%2f%2fmedia2.giphy.com%2fmedia%2fl4JyZuXNGxS3Yydeo%2fgiphy.gif%3fcid%3d790b7611318eb5b864ad67b3cecb35b9d81240a50d251bb0%26rid%3dgiphy.gif%26ct%3dg&ehk=A6wfp3Afm3rFCPLWSjgQd6JVjmRSBNBlk9vd0jVNgJc%3d ",
    "clown":"https://th.bing.com/th/id/R.4c8321133c320bd772fe90c80fed3133?rik=OPrV3%2b1Zkelr2A&pid=ImgRaw&r=0 ",
    "cow" : "https://lifeprint.com/asl101/gifs/c/cow.gif ",
    "cry" : "https://www.lifeprint.com/asl101/gifs/c/cry-tears.gif ",
    "cute" : "https://lifeprint.com/asl101/gifs/c/cute-sugar.gif ",
    "dad" : "https://media.giphy.com/media/l0MYQcLKwtl5v6H1S/giphy.gif ",
    "dance":"https://www.bing.com/th/id/OGC.b0dffc2302a0a29d029bf11ae004d430?pid=1.7&rurl=https%3a%2f%2fmedia.giphy.com%2fmedia%2f3o7TKMspYQjQTbOz2U%2fgiphy.gif&ehk=h%2bdBHCxuoOT89ovSy5uTk6MCL9acaBEV6ld9lrVDRF4%3d ",
    "dirty":"https://th.bing.com/th/id/OIP.wRA7r1OPPUuEoLL4Hds9jAHaHa?rs=1&pid=ImgDetMain ",
    "dog":"https://th.bing.com/th/id/R.60bf4344ac442956c1c5e95753b9e7e4?rik=uVshGsOHXgldoQ&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fd%2fdog1.gif&ehk=Xnfrvg0xwy1u%2fae9vWdKYMT25%2bF6qoteW2FThCbVrOA%3d&risl=&pid=ImgRaw&r=0 ",
    "doll":"https://www.bing.com/th/id/OGC.295d59735f136734f6162128faf607fc?pid=1.7&rurl=https%3a%2f%2fwww.lifeprint.com%2fasl101%2fgifs-animated%2fdoll.gif&ehk=hPI0Fzzl9CGOrgQYS2Z53a5YdYgjxYFeOIGghGAEZYU%3d ",
    "donkey" : "https://www.lifeprint.com/asl101/gifs/d/donkey-1h.gif ",
    "drink" : "https://www.lifeprint.com/asl101/gifs/d/drink-c.gif ",
    "duck":"https://th.bing.com/th/id/R.d0ff898c2a7d366d99b90bc6b15b4da6?rik=ZetjiJ3WOhOXrQ&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fd%2fduck.gif&ehk=STeui62x5lieai0VcyeZkX2t8rILR%2f8GR5F3x2xJ5tw%3d&risl=&pid=ImgRaw&r=0 ",
    "ear" : "https://lifeprint.com/asl101/signjpegs/e/ears.h3.jpg ",
    "elephant" : "https://lifeprint.com/asl101/gifs-animated/elephant.gif ",
    "empty" : "https://lifeprint.com/images-signs/empty.gif ",
    "every" : "https://lifeprint.com/asl101/gifs-animated/every.gif ",
    "eye" : "https://lifeprint.com/asl101/gifs/e/eyes.gif ",
    "face" : "https://lifeprint.com/asl101/gifs/f/face.gif ",
    "fine":"https://th.bing.com/th/id/R.fa49db1ccf5745e5999768dc5f24d14b?rik=Qpm%2bw3fHTAWj1A&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2ff%2ffine.gif&ehk=mGMZf4l%2bLZMq4atRomNJSvrSjYgFe%2bRVCm1dYLh5J3I%3d&risl=&pid=ImgRaw&r=0 ",
    "finish":"https://th.bing.com/th/id/R.6f093bc5abdd108a9a58bc45e8b785eb?rik=34j4pW2f3E5TtQ&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2ff%2ffinish.gif&ehk=xNk24Jbe3t0moSmcmUftmZzCRgHIxsarq3W9E7kGmPM%3d&risl=&pid=ImgRaw&r=0 ",
    "fireman" : "https://lifeprint.com/asl101/gifs/f/fireman-c2.gif ",
    "first" : "https://lifeprint.com/asl101/gifs/f/first.gif ",
    "fish":"https://th.bing.com/th/id/OIP.Lzhd7lIIa-V4H3faS1d3mQHaHa?rs=1&pid=ImgDetMain ",
    "flag":"https://th.bing.com/th/id/OIP.3LqQWEnK4TG0lohgQ3G5uAHaE-?w=263&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7 ",
    "flower" : "https://media.giphy.com/media/3o7TKGkqPpLUdFiFPy/giphy.gif ",
    "food":"https://i.pinimg.com/originals/cc/bb/0c/ccbb0c143db0b51e9947a5966db42fd8.gif ",
    "frenchfries" : "https://www.lifeprint.com/asl101/gifs/f/french-fries.gif ",
    "frog" : "https://media.giphy.com/media/l0HlKl64lIvTjZ7QA/giphy.gif ",
    "garbage":"https://th.bing.com/th/id/R.6f9fbc2551cb95a2bfe8b1e7a2e9d2d2?rik=78iU%2fDx85Ut9fA&riu=http%3a%2f%2fwww.lifeprint.com%2fasl101%2fgifs%2fg%2fgarbage.gif&ehk=lafY%2f1y5WEEfr04p6Uq4waDP9iV7bJB5r2k3RYGOhWY%3d&risl=&pid=ImgRaw&r=0 ",
    "gift" : "https://www.babysignlanguage.com/signs/gift.gif ",
    "giraffe" : "https://www.lifeprint.com/asl101/gifs/g/giraffe.gif ",
    "girl":"https://th.bing.com/th/id/R.a66fd1d681a94a4c783bb4e3b21816e8?rik=yDsGUPEaDyeSlA&riu=http%3a%2f%2fwww.babysignlanguage.com%2fsigns%2fgirl.gif&ehk=zdVxVSayRBDn67vVCpMhUH6UmzUQE8vaY7%2bv8jedvs8%3d&risl=&pid=ImgRaw&r=0 ",
    "give" : "https://www.lifeprint.com/asl101/gifs/g/give-x-two-handed.gif ",
    "go" : "https://media.giphy.com/media/l3vRdVMMN9VsW5a0w/giphy.gif ",
    "goose" : "https://www.babysignlanguage.com/signs/goose.gif ",
    "grandma" : "https://www.lifeprint.com/asl101/gifs/g/grandma.gif ",
    "grandpa":"https://th.bing.com/th/id/OIP.yyLPc-rWg0PMNbrwjeQQngHaE-?w=238&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7 ",
    "grass":"https://th.bing.com/th/id/R.bb08e8234c86c4c2eb4703b681e337aa?rik=uGZNVzt6tISwHA&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs-animated%2fgrass.gif&ehk=VKQd9cvBrglo47EhogWYL9rOiZZsEJ7Yqt%2bgJ8N99yQ%3d&risl=&pid=ImgRaw&r=0 ",
    "green":"https://i.pinimg.com/originals/cb/7f/75/cb7f757ffb79cb3d1309c9ad785e83a1.gif ",
    "hair" : "https://www.lifeprint.com/asl101/gifs/h/hair-g-version.gif ",
    "happy":"https://media0.giphy.com/media/3o7TKFpahYpUp4g0N2/giphy.gif?cid=790b7611bca188a92e2e759876756ef62ba95b7cdd337c77&rid=giphy.gif&ct=g ",
    "hate" : "https://media.giphy.com/media/l0MYPiNw8l2LAPJXW/giphy.gif ",
    "have":"https://th.bing.com/th/id/R.ca7e64fe551ff1d05246ea80b93ab08d?rik=q5Ei%2b7oJb7Uzyw&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fh%2fhave.gif&ehk=H9yIaJxFVejkfHpkhTUipBRv9CW63KBFy6QW5cdbkKw%3d&risl=&pid=ImgRaw&r=0 ",
    "head":"https://th.bing.com/th/id/R.a01fc1c3b6d04dce1e25b5ba484fa5cb?rik=OcbJdRbpEFsWXQ&riu=http%3a%2f%2flifeprint.com%2fasl101%2fsignjpegs%2fh%2fhead-1.jpg&ehk=RPBV45fSrLDEWYiZvRuZs2c1JNrL4WzdqLSNMFIF3Rs%3d&risl=&pid=ImgRaw&r=0 ",
    "hear" : "https://www.lifeprint.com/asl101/signjpegs/h/hear.h4.jpg ",
    "helicopter":"https://th.bing.com/th/id/R.b191ff68d4d2a367caf83574acfd7d22?rik=5uhW2xBaByliWA&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fh%2fhelicopter.gif&ehk=mwAyT82RBoeYDe7yaHA1jL3%2f30dUksltmv4dF7YGf%2bU%3d&risl=&pid=ImgRaw&r=0 ",
    "hello" : "https://media.giphy.com/media/3o7TKNKOfKlIhbD3gY/giphy.gif ",
    "hen":"https://media0.giphy.com/media/26hisADhtILiu1J3W/giphy.gif?cid=790b76112d512b94e1647afb111c8d77f92ae31f37864f2&rid=giphy.gif&ct=g ",
    "home":"https://th.bing.com/th/id/R.74bcbf6b29b0f5aa89edcc6d62980040?rik=%2bnBd%2foQjxnoPfg&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fh%2fhome-2.gif&ehk=7yD%2f%2fh6JN1Y4D4BOrUjgKW4Jccy2Y4GVYLf%2fzyk%2b5YY%3d&risl=&pid=ImgRaw&r=0 ",
    "horse" : "https://media.giphy.com/media/l0HlM5HffraiQaHUk/giphy.gif ",
    "hot" : "https://media.giphy.com/media/3o6Zt99k5aDok347bG/giphy.gif ",
    "hungry" : "https://media.giphy.com/media/l3vR0xkdFEz4tnfTq/giphy.gif ",
    "icecream" : "https://media.giphy.com/media/3o7TKp6yVibVMhBSLu/giphy.gif ",
    "jacket" : "https://www.lifeprint.com/asl101/gifs/c/coat.gif ",
    "jump" : "https://lifeprint.com/asl101/gifs-animated/jump.gif ",
    "kiss" : "https://i.gifer.com/PxGY.gif ",
    "kitty" : "https://lifeprint.com/asl101/gifs-animated/cat-02.gif ",
    "later":"https://media3.giphy.com/media/l0MYHTyMzMRcikIxi/giphy.gif?cid=790b761128cd39f9baa06dbeb4e099d13e3516763d5f0952&rid=giphy.gif&ct=g ",
    "like" : "https://lifeprint.com/asl101/gifs/l/like.gif ",
    "look":"https://th.bing.com/th/id/R.60c1c51f561769b1cfea7b35403c0cb0?rik=pYhzip7LqNs7qw&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fl%2flook-at-1.gif&ehk=rFJ7dBrMGFDK0nHLzrOPAzROVE7yqyDEcb%2btLqKqYOI%3d&risl=&pid=ImgRaw&r=0 ",
    "loud" : "https://lifeprint.com/asl101/gifs-animated/loud.gif ",
    "mad" : "https://lifeprint.com/asl101/gifs/m/mad.gif ",
    "man" : "https://lifeprint.com/asl101/gifs/m/man.gif ",
    "many" : "https://lifeprint.com/asl101/gifs/m/many.gif ",
    "milk" : "https://lifeprint.com/asl101/gifs/m/milk.gif ",
    "mitten" : "https://lifeprint.com/asl101/gifs-animated/mittens.gif ",
    "mom" : "https://lifeprint.com/asl101/gifs/m/mom.gif ",
    "moon":"https://th.bing.com/th/id/R.17741dd479443ddcdc068702a0235e9b?rik=XbVhBJtkANrG9g&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fm%2fmoon.gif&ehk=YSDvFeUSTa9X1BEJhDjdnLC4c7zWn8z7Hj%2fMkkLUyFE%3d&risl=&pid=ImgRaw&r=0 ",
    "morning" : "https://media0.giphy.com/media/3o6ZtrcJ9GCXGGw0ww/source.gif ",
    "mouse" : "https://lifeprint.com/asl101/gifs/m/mouse.gif ",
    "mouth" : "https://lifeprint.com/asl101/gifs-animated/mouth.gif ",
    "night" : "https://lifeprint.com/asl101/gifs/n/night.gif ",
    "no" : "https://lifeprint.com/asl101/gifs/n/no-2-movement.gif ",
    "nose" : "https://lifeprint.com/asl101/signjpegs/n/nose.h1.jpg ",
    "not":"https://th.bing.com/th/id/R.d952267a4c8d69033281d8fdc15e4597?rik=6%2bbZ2jRA%2famQ4Q&riu=http%3a%2f%2flifeprint.com%2fasl101%2fgifs%2fn%2fnot-negative.gif&ehk=%2bppuO9P0%2fpdzrrdNO4FXpxdIGs8jgY%2fj%2b1ZCwdbDWO4%3d&risl=&pid=ImgRaw&r=0 ",
    "now" : "https://lifeprint.com/asl101/gifs/n/now.gif ",
    "old" : "https://lifeprint.com/asl101/gifs/o/old.gif ",
    "on" : "https://lifeprint.com/asl101/gifs/o/on-onto.gif ",
    "orange" : "https://lifeprint.com/asl101/gifs/o/orange.gif ",
    "owl" : "https://lifeprint.com/asl101/gifs/o/owl.gif ",
    "pencil" : "https://lifeprint.com/asl101/gifs/p/pencil-2.gif ",
    "pig" : "https://lifeprint.com/asl101/gifs/p/pig.gif ",
    "pizza" : "https://lifeprint.com/asl101/gifs/p/pizza.gif ",
    "please" : "https://lifeprint.com/asl101/gifs-animated/pleasecloseup.gif ",
    "police":"https://th.bing.com/th/id/R.7a8d1261d549a35a3e703a6371d84948?rik=icjjfUg15cqgLw&pid=ImgRaw&r=0 ",
    "pretty" : "https://lifeprint.com/asl101/gifs/b/beautiful.gifm ",
    "quiet" : "https://lifeprint.com/asl101/gifs-animated/quiet-03.gif ",
    "rain" : "https://lifeprint.com/asl101/gifs/r/rain.gif ",
    "read" : "https://lifeprint.com/asl101/gifs/r/read.gif ",
    "red" : "https://lifeprint.com/asl101/gifs-animated/red.gif ",
    "refrigerator" : "https://lifeprint.com/asl101/gifs/r/refrigerator-r-e-f.gif ",
    "ride" : "https://lifeprint.com/asl101/gifs/r/ride.gif ",
    "room" : "https://lifeprint.com/asl101/gifs/r/room-box.gif ",
    "sad" : "https://lifeprint.com/asl101/gifs/s/sad.gif ",
    "same" : "https://lifeprint.com/asl101/gifs/s/same-similar.gif ",
    "see" : "https://lifeprint.com/asl101/gifs/l/look-at-2.gif ",
    "shhh" : "https://lifeprint.com/asl101/signjpegs/s/shhh.jpg ",
    "shirt" : "https://lifeprint.com/asl101/gifs/s/shirt-volunteer.gif ",
    "shoe" : "https://media.giphy.com/media/3o7TKC4StpZKa6d2y4/giphy.gif ",
    "shower" : "https://lifeprint.com/asl101/gifs/s/shower.gif ",
    "sick" : "https://lifeprint.com/asl101/gifs/s/sick.gif ",
    "sleep":"https://media4.giphy.com/media/3o7TKnRuBdakLslcaI/200.gif?cid=790b76110d8f185a9713f36dd65a0df801576e01b403c95c&rid=200.gif&ct=g ",
    "sleepy":"https://th.bing.com/th/id/R.1770287a186a5c7b444339b0fbab22ed?rik=zdWvzvABcDHTdw&riu=http%3a%2f%2fwww.lifeprint.com%2fasl101%2fgifs-animated%2fsleepy.gif&ehk=zLqDFJMAs2nqG02RbbR6mEMvux4h85JGzls4uwgrePQ%3d&risl=&pid=ImgRaw&r=0 ",
    "snack" : "https://media.giphy.com/media/26ybw1E1GTKzLuKDS/giphy.gif ",
    "snow" : "https://lifeprint.com/asl101/gifs/s/snow.gif ",
    "stay" : "https://i.pinimg.com/originals/f5/29/8e/f5298eaa46b91cd6de2a32bd76aadffc.gif ",
    "stuck" : "https://lifeprint.com/asl101/signjpegs/s/stuck.2.jpg ",
    "sun" : "https://media.giphy.com/media/3o6Zt7merN2zxEtNRK/giphy.gif ",
    "table" : "https://lifeprint.com/asl101/gifs/t/table.gif ",
    "talk" : "https://lifeprint.com/asl101/gifs/t/talk.gif ",
    "taste" : "https://lifeprint.com/asl101/gifs/t/taste.gif ",
    "thankyou" : "https://lifeprint.com/asl101/gifs/t/thank-you.gif ",
    "there" : "https://lifeprint.com/asl101/gifs-animated/there.gif ",
    "think" : "https://lifeprint.com/asl101/gifs/t/think.gif ",
    "thirsty" : "https://media.giphy.com/media/l3vR0sYheBulL1P7W/giphy.gif ",
    "tiger" : "https://lifeprint.com/asl101/gifs/t/tiger.gif ",
    "time" : "https://lifeprint.com/asl101/gifs/t/time-1.gif ",
    "tomorrow" : "https://lifeprint.com/asl101/gifs/t/tomorrow.gif ",
    "tooth":"https://th.bing.com/th/id/R.c47dfb217103be17e854132a61bbc232?rik=ZF%2fsFUXvt5czGA&riu=http%3a%2f%2flifeprint.com%2fasl101%2fsignjpegs%2ft%2fteeth1.jpg&ehk=vI5eDlD4HZWXhK1PQOQz4nA5e6oguHgeXqDo%2fcdcWg4%3d&risl=&pid=ImgRaw&r=0 ",
    "toothbrush":"https://www.bing.com/th/id/OGC.249e98487b35c24226c15faf2abafe07?pid=1.7&rurl=https%3a%2f%2fmedia.giphy.com%2fmedia%2fl3vR0Rq2HVL2KHLUI%2fgiphy.gif&ehk=eC0Sq9sHjrrOrkyJvOogQbXVkTOL5OPCeyVymejL0RU%3d ",
    "toy" : "https://lifeprint.com/asl101/gifs-animated/play-02.gif ",
    "tree" : "https://lifeprint.com/asl101/gifs-animated/tree.gif ",
    "uncle" : "https://lifeprint.com/asl101/gifs/u/uncle.gif ",
    "underwear":"https://th.bing.com/th/id/OIP.c8g9T_lOhbZWRvKAA12J8wHaEO?pid=ImgDet&w=310&h=177&rs=1 ",
    "wait" : "https://lifeprint.com/asl101/gifs/w/wait.gif ",
    "wake" : "https://lifeprint.com/asl101/gifs/w/wake-up.gif ",
    "water" : "https://lifeprint.com/asl101/gifs/w/water-2.gif ",
    "wet" : "https://www.babysignlanguage.com/signs/wet.gif ",
    "where" : "https://lifeprint.com/asl101/gifs/w/where.gif ",
    "white" : "https://lifeprint.com/asl101/gifs/w/white.gif ",
    "who" : "https://lifeprint.com/asl101/gifs/w/who.gif ",
    "why" : "https://lifeprint.com/asl101/gifs/w/why.gif ",
    "will" : "https://lifeprint.com/asl101/gifs/f/future.gif ",
    "wolf" : "https://lifeprint.com/asl101/gifs/w/wolf-side-view.gif ",
    "yellow" : "https://lifeprint.com/asl101/gifs/y/yellow.gif ",
    "yes" : "https://media.tenor.com/oYIirlyIih0AAAAC/yes-asl.gif ",
    "yesterday" : "https://lifeprint.com/asl101/gifs/y/yesterday.gif ",
    "zebra" : "https://lifeprint.com/asl101/gifs/z/zebra-stripes-two-hands.gif ",
    "Zipper":"https://th.bing.com/th/id/R.cacf57bbb2001f13e223f45329db5ecb?rik=qPRTVGd2SzUBxw&riu=http%3a%2f%2fwww.babysignlanguage.com%2fsigns%2fzipper.gif&ehk=IGx68sSokNwU21zu3Z2D%2blmeehKYxpSNhX2VnrvQqYE%3d&risl=&pid=ImgRaw&r=0 "
}

# Create the dropdown list
selected_word = st.selectbox("Select a word", list(word_to_image.keys()))

# Display the corresponding image
if selected_word:
    image_file = word_to_image[selected_word]
    st.markdown(f"![{selected_word.capitalize()}]({image_file})", unsafe_allow_html=True)


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
        current_word_text.text(f'Current predicted value: {word}')
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
        current_sentence_text.text(f'Current sentence : {sentence_text}')
        # Show to screen
        if is_recording:
            # Display the video feed only when recording is active
            st_image_placeholder.image(image, channels="BGR")
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
