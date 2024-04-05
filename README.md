# BUILD
Sign Language Project for INIT Build
## Running the Streamlit App in a Docker Container

1. **Build Docker Image:**
   Run `docker build -t my-streamlit-app .` in the terminal.

2. **Run Docker Container:**
   Execute `docker run -p 8501:8501 --privileged my-streamlit-app`.

3. **Access the Streamlit App:**
   Visit `http://localhost:8501` in your web browser.

4. **Interacting with the App:**
   Interact with your Streamlit app as usual.

5. **Stopping the Container:**
   Stop the Docker container using `Ctrl+C` or Docker CLI.

