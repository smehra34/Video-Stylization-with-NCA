
import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import torchvision.models as torch_models
import numpy as np


class ContentLoss(torch.nn.Module):
    def __init__(self, args):
        super(ContentLoss, self).__init__()

        self.args = args
        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(args.DEVICE)

        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, target_images, generated_images):
        
        loss = 0.0
    
        with torch.no_grad():
            target_features = get_content_features_vgg(self.args, target_images, self.vgg16)
        generated_features = get_content_features_vgg(self.args, generated_images, self.vgg16)

        loss += self.loss_fn(target_features, generated_features)

        return loss


def get_content_features_vgg(args, imgs, vgg_model):

    DEVICE = args.DEVICE

    content_layer = 19 # conv4_2 (if want output after relu, use 20)
    
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(DEVICE)
    x = (imgs - mean) / std

    for i, layer in enumerate(vgg_model[:content_layer + 1]):
        x = layer(x)
    return x
    