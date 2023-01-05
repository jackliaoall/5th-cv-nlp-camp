import flask
import io
import threading
import json
import base64
import numpy as np
import cv2
from models import *
from utils.utils import *
from utils.datasets import *
 
import os
import sys
import time
import datetime
import argparse
 
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
app = flask.Flask(__name__)

use_cuda = torch.cuda.is_available()

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    #配置文件和权重文件路径
    config_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\config\\yolov3.cfg'
    weights_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\weights\\yolov3.weights'
    
    class_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\data\\coco.names'
    global classes
    classes = load_classes(class_path)
    
    global model
    #默认网络输入大小为416
    model = Darknet(config_path)
    #载入模型
    model.load_darknet_weights(weights_path)
    if use_cuda:
        model.cuda() 
    model.eval()
 
class_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\data\\coco.names'
classes = load_classes(class_path)
 
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
 

def np2tensor(np_array, img_size):
    h, w, _ = np.array(np_array).shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    img_shape = (img_size, img_size)
    input_img = np.pad(np_array, pad, 'constant', constant_values=127.5) / 255.
    input_img = cv2.resize(input_img, (img_size,img_size))
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = torch.from_numpy(input_img).float()
 
    return input_img
 
 
def yolo_detection(img_array, img_size = 416):
    img_array = np.array(img_array)
    img_tensor = np2tensor(img_array,img_size)
    img_tensor = Variable(img_tensor.type(Tensor))
    img_tensor = img_tensor.unsqueeze(0)
 
    
    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections)
    
    pad_x = max(img_array.shape[0] - img_array.shape[1], 0) * (img_size / max(img_array.shape))
    pad_y = max(img_array.shape[1] - img_array.shape[0], 0) * (img_size / max(img_array.shape))    
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
 
    results=[]
    if detections is not None:
        detection = detections[0]
        unique_labels = detection[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            box_h = ((y2 - y1) / unpad_h) * img_array.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img_array.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img_array.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img_array.shape[1]
 
            class_name = classes[int(cls_pred)]
            detect_result ={'class':class_name, 'x':x1.item(), 'y':y1.item(), 'h':box_h.item(), 'w':box_w.item()}
            results.append(detect_result)
        
    
    data_json = json.dumps(results,sort_keys=True, indent=4, separators=(',', ': '))
 
    return data_json


@app.route("/predict", methods=["POST"])
def predict():
    
    data = {"success": False}
    
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            
            image = flask.request.files["image"].read()
            
            image = Image.open(io.BytesIO(image)).convert('RGB')
                     
            res = yolo_detection(image)      
            
            data['predictions'] = res
            
            data["success"] = True
        
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()
