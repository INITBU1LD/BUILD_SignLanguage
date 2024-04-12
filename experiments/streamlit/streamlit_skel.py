# Streamlit WebRTC skeleton code
import streamlit as st
from experiments.streamlit.streamlit_skel import VideoTransformerBase, webrtc_streamer

# Define a custom video transformer to display webcam stream
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # You can perform any image processing here
        return frame

def main():
    st.title("Webcam Stream Example")

    # Start webcam stream using streamlit-webrtc
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    # Display error message if webcam is not available
    if not webrtc_ctx.state.playing:
        st.error("Error: Unable to open webcam.")

if __name__ == "__main__":
    main()
