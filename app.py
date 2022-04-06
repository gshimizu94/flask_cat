from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from datetime import datetime
import os
import string
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms, models 
from torchvision.datasets import ImageFolder
import flask_login

#モデルの定義
class DCResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 1)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = self.out(x)
        return x

def resize(img, shape=(64, 64)):
    return cv2.resize(img, shape)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = transforms.Compose([
            np.array,
            resize,
    ])
    img = trans(img)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
    return img

def load_model():
    global model
    print(" * Loading pre-trained model ...")
    model = DCResnet()
    model.load_state_dict(torch.load('best_resnet.pth', map_location=torch.device('cpu')))
    model.eval()
    print(' * Loading end')
    return model

app = Flask(__name__)

@app.route('/')
def index():
    load_model()
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        model = load_model()
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        img = preprocess(img)
        predict_Confidence = round(model(img.unsqueeze(0)).item()*100, 1)
        predict_Confidence = str(predict_Confidence)
        # render_template('./result.html')
  
        return render_template('./result.html', title='ネコ度', predict_Confidence=predict_Confidence)

if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run()