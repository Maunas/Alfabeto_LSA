import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import matplotlib.pyplot as plt

class CustomBoxPredictor(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomBoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        # Agregamos una capa para la regresión de las cajas
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x):
        # Retornamos los logits de las clases y las regresiones de las cajas
        return self.cls_score(x), self.bbox_pred(x)

def load_model():
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 27 #Letras + fondo
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = CustomBoxPredictor(in_features, num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load('prod/modelo.pt',weights_only=True, map_location=device))
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
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

  # Ejecuta el modelo en la imagen y obtén las predicciones
  model.eval()
  with torch.no_grad():
      detections = model([image.to(device)])[0]

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
      # Etiqueta con la clase y el puntaje de confianza
      label = f"Class: {get_letter_from_number(labels[i].item())}, Score: {scores[i].item():.2f}"
      plt.text(xmin, ymin, label, color='white', fontsize=10,
              bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
  plt.axis('off')
  return plt

