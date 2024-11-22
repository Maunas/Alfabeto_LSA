import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import requests
import os

def download_model_from_drive(destination):
    # URL para descargar archivos compartidos de Google Drive
    download_url = "https://drive.google.com/uc?id=1ZDyu6xUNLlo8s3SL4e0AhDQpWNvicNr8"
    
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Modelo guardado en {destination}.")
    else:
        print("Error al descargar el archivo.")
        response.raise_for_status()

ruta_modelo = "prod/modelo.pt"
if not os.path.exists(ruta_modelo):
    download_model_from_drive(ruta_modelo)

class CustomBoxPredictor(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomBoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        # Agregamos una capa para la regresión de las cajas
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x):
        # Retornamos los logits de las clases y las regresiones de las cajas
        return self.cls_score(x), self.bbox_pred(x)

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 27 #Letras + fondo
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = CustomBoxPredictor(in_features, num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    #Cambiada la ruta, por superar los límites de GLF
    #model.load_state_dict(torch.load('prod/modelo.pt',weights_only=True , map_location=device))

    model.load_state_dict(torch.load(ruta_modelo
        ,weights_only=True , map_location=device))

    model.eval()
    return model, device

label_map = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9,
    "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17,
    "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25,
    "Z": 26
}

inverse_label_map = {v: k for k, v in label_map.items()}

def get_letter_from_number(number):
    return inverse_label_map.get(number, "Unknown")  # Retorna "Unknown" si el número no está en el mapa

def predict_and_plot_image(model, image, device):
  imagePIL = Image.open(image).convert("RGB")
  transform = T.ToTensor()
  image_tensor = transform(imagePIL).to(device)

  # Ejecuta el modelo en la imagen y obtén las predicciones
  model.eval()
  with torch.no_grad():
      detections = model([image_tensor.to(device)])[0]

  boxes = [detections['boxes'][0]]
  labels = [detections['labels'][0]]
  scores = [detections['scores'][0]]

  # Graficar la imagen y las cajas
  plt.figure(figsize=(12, 8))
  plt.imshow(imagePIL)
  for i, box in enumerate(boxes):
      xmin, ymin, xmax, ymax = box.cpu()
      plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        edgecolor='green', fill=False, linewidth=2))

      label = f"Class: {get_letter_from_number(labels[0].item())}, Score: {scores[i].item():.2f}"
      plt.text(xmin, ymin, label, color='white', fontsize=10,
              bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
  plt.axis('off')
  return convert_plot_to_img(plt), get_letter_from_number(labels[0].item())

def convert_plot_to_img(plot):
    # Convertir el gráfico a una imagen en formato PNG
    buf = BytesIO()
    plot.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)  # Volver al principio del buffer para poder leerlo
    return buf