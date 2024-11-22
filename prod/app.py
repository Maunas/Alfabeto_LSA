import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 

model, device = load_model()
gallery = []

st.title("Detector de alfabeto LSA")
st.write("Sacate una foto realizando una seña del alfabeto, o sube una foto al sistema.")
image_camera = st.camera_input("Saca una foto de la seña")
image_file = st.file_uploader("Sube una foto", type=["png", "jpg"])

# Check if an image is uploaded before attempting to process it
if image_camera is not None :
    plot = predict_and_plot_image(model, image_camera, device)
    st.pyplot(plot)
    gallery.append(plot)

if image_file is not None :
    plot = predict_and_plot_image(model, image_file, device)
    st.pyplot(plot)
    gallery.append(plot)