import cv2
import torch
from super_gradients.training import  models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

modle = model.to('cuda' if torch.cuda.is_available() else 'cpu')

modle.predict_webcam()