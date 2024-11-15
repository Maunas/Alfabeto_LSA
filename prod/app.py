import torch
import streamlit as st
import load_model, predict_and_plot_image, label_map from utils

model, device = load_model()
image = st.file_uploader("Sube una foto", type=["png","jpg"])
plot = predict_and_plot_image(model, image, device, label_map)
st.pyplot(plot)