
import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import torchvision.models as torch_models
import numpy as np


class ContentLoss(torch.nn.Module):
    def __init__(self, device):
        super(ContentLoss, self).__init__()

        self.device = device
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(device)

    def forward(self, input_dict):
        loss = 0.0
        target_images = input_dict['target_images']
        generated_images = input_dict['generated_images']

        b, c, h, w = generated_images.shape
        _, _, ht, wt = target_images.shape

        # # Scale the images before feeding to VGG
        # generated_images = (generated_images + 1.0) / 2.0
        # target_images = (target_images + 1.0) / 2.0

        if h != ht or w != wt:
            target_images = TF.resize(target_images, size=(h, w))

        with torch.no_grad():
            target_features = get_content_feature_vgg(self.device, target_images, self.vgg16)
        generated_features = get_content_feature_vgg(self.device, generated_images, self.vgg16)

        loss += self.loss_fn(target_features, generated_features)

        return loss



def get_content_feature_vgg(device, imgs, vgg_model):

    content_layer = 19 # conv4_2 (if want output after relu, use 20)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(device)
    x = (imgs - mean) / std

    for i, layer in enumerate(vgg_model[:content_layer + 1]):
        x = layer(x)
    return x
