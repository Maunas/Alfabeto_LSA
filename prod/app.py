import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 
from streamlit_extras.grid import grid

st.set_page_config(page_title="Alfabeto LSA")

model, device = load_model()

# Inicializar galería en la sesión
if "gallery" not in st.session_state:
    st.session_state["gallery"] = []

st.title("Detector de alfabeto LSA")
st.write("Sacate una foto realizando una seña del alfabeto, o sube una foto al sistema.")

modo = st.radio("Elegir Método de Captura", [
    "Abrir Cámara", "Subir Foto", "Galería"
], index=0,horizontal=True, )

if modo == "Abrir Cámara":
    image = st.camera_input("Saca una foto de la seña")
    if image is not None :
        plot = predict_and_plot_image(model, image, device)
        st.pyplot(plot)
        st.session_state["gallery"].append(plot)

elif modo == "Subir Foto":
    image = st.file_uploader("Sube una foto", type=["png", "jpg"])
    if image is not None :
        plot = predict_and_plot_image(model, image, device)
        st.pyplot(plot)
        st.session_state["gallery"].append(plot)

# Modo: Galería
elif modo == "Galería":
    if st.session_state["gallery"]:
        my_grid = grid(5, "small")
    else:
        st.write("Aún no hay imágenes en la galería.")