import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 
from PIL import Image

model, device = load_model()
image = st.file_uploader("Sube una foto", type=["png","jpg"])
# Check if an image is uploaded before attempting to process it
if image is not None:
    imagePIL = Image.open(image)
    plot = predict_and_plot_image(model, imagePIL, device)
    st.pyplot(plot)
else:
    st.write("Sube una foto.")