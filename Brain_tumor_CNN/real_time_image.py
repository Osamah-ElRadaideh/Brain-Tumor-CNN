import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from dataset import BrainTumor
import cv2
import lazy_dataset
from utils import collate
from model import Network
from einops import einops
from tqdm import tqdm
import torch.nn as nn
import os 
from pathlib import Path
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_example(example):
    path = example['file_path']
    img = cv2.imread(path,0)
    img = cv2.resize(img,(256,256))
    example['image'] = img
    return example



model = Network().to(device=device)
PATH = torch.load('D:\\trained_models\\image_classification\model.pth')
model.load_state_dict(PATH)
model.eval()
with torch.no_grad():
    image_path = input('input image path: ')
    print(image_path)
    image = cv2.imread(image_path,0)
    print(image.shape)
    image = cv2.resize(image,(256,256))
    image = einops.rearrange(image,'b f -> 1 1 b f')
    image = torch.Tensor(image).to(device=device)
    output = model(image)
    probs = F.softmax(output)[0]
    probs = torch.mul(probs,100)
    print(f'classification probabilities:')
    print(f'glioma: {probs[0]}',f'meningioma: {probs[1]}',f'no tumor: {probs[2]}',f'pituitary tumor: {probs[3]}',sep='\n')
