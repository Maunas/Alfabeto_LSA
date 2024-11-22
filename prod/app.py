import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 

model, device = load_model()
gallery = []


st.title("Detector de alfabeto LSA")
st.write("Sacate una foto realizando una seña del alfabeto, o sube una foto al sistema.")

modo = st.radio("Elegir Método de Captura", [
    "Abrir Cámara", "Subir Foto"
], index=0,horizontal=True, )

if modo == "Abrir Cámara":
    image = st.camera_input("Saca una foto de la seña")
    if image is not None :
        plot = predict_and_plot_image(model, image, device)
        st.pyplot(plot)
        gallery.append(plot)

if modo == "Subir Foto":
    image_file = st.file_uploader("Sube una foto", type=["png", "jpg"])
    if image_file is not None :
        plot = predict_and_plot_image(model, image_file, device)
        st.pyplot(plot)
        gallery.append(plot)


    