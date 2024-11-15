import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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
    model.load_state_dict(torch.load('./modelo.pt',weights_only=True, map_location=device))
    model.eval()
    return model, device