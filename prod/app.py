import torch
import streamlit as st
from utils import load_model, predict_and_plot_image 
from streamlit_extras.grid import grid

st.set_page_config(page_title="Alfabeto LSA")

model, device = load_model()
image = None

# Inicializar galería en la sesión
if "gallery" not in st.session_state:
    st.session_state["gallery"] = []

def predict_and_save(image):
    plot, label = predict_and_plot_image(model, image, device)
    st.image(plot)
    if st.button("Guardar en Galería"):
        st.session_state["gallery"].append((plot, label))
        st.success("Imagen guardada en la galería.")

#-----------------------Pagina---------------------#

st.title("Detector de alfabeto LSA")
st.write("Sacate una foto realizando una seña del alfabeto, o sube una foto al sistema.")

modo = st.radio("Elegir Método de Captura", [
    "Abrir Cámara", "Subir Foto", "Galería"
], index=0,horizontal=True, )

if modo == "Abrir Cámara":
    image = st.camera_input("Saca una foto de la seña")
    if image is not None :
        predict_and_save(image)

elif modo == "Subir Foto":
    image = st.file_uploader("Sube una foto", type=["png", "jpg"])
    if image is not None :
        predict_and_save(image)

# Modo: Galería
elif modo == "Galería":
    if st.session_state["gallery"]:
        # Recorremos las imágenes en la galería
        columns = st.columns(5)
        for i, (img, label) in enumerate(st.session_state["gallery"]):
            col = columns[i % 5]  # Repartimos las imágenes entre las columnas
            with col:
                st.image(img)
                st.write(label)
    else:
        st.write("Aún no hay imágenes en la galería.")