import streamlit as st
import time
import os
import subprocess
import torch
from transformers import pipeline

# Title and Subheader
st.title("Sentiment Analysis Web App")
st.subheader("Welcome to the Sentiment Analysis Tool")

#-------------------------------------------------------------------------------
# Function to Clone the GitHub Repository
def clone_repo():
    repo_url = "https://github.com/yourusername/yourrepo.git"  # Replace with your repo URL
    repo_dir = "ml-model"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url])
        print(f"Repository cloned to {repo_dir}")
    else:
        print(f"Repository already exists at {repo_dir}")

# Clone the repository to get the model
clone_repo()

#-------------------------------------------------------------------------------
# Initialize Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Path to the model (adjust based on your repo structure)
model_path = 'ml-model/ml_model'  # Update the path if needed
classifier = pipeline('text-classification', model=model_path, device=device)

# Input text box
user_text = st.text_area(label="Please enter text below...", label_visibility="visible")

# Analyze and clear buttons
col1, col2 = st.columns(2)
analyze_button = col1.button("Analyze Sentiment")
clear_button = col2.button("Clear")

# Handle Sentiment Analysis when analyze button is clicked
if analyze_button:
    if user_text:
        with st.spinner("Working on the input..."):
            time.sleep(0.5)
            st.toast("Analysis completed")
            time.sleep(0.5)
            st.info("Sentiment analysis complete. 👇👇👇")
            time.sleep(0.5)
            output = classifier(user_text)
            st.write(output)
    else:
        st.warning("I am speechless here 😶. Please provide some text to analyze sentiment.")
        st.image("https://th.bing.com/th/id/OIP.xqHR0bxoLZeRA1xO_xM41wHaFx?rs=1&pid=ImgDetMain", width=200)

# Clear button functionality
if clear_button:
    st.rerun()

