import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 

model, device = load_model()
image = st.file_uploader("Sube una foto", type=["png","jpg"])
plot = predict_and_plot_image(model, image, device)
st.pyplot(plot)