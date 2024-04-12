'''
Modified from base implementation in:
    Niklasson, E., Mordvintsev, A., Randazzo, E., & Levin, M. (2021).
    "Self-organising textures"

https://distill.pub/selforg/2021/textures/
https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb
'''


import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms


class StyleLoss():

    def __init__(self, target_style_image, num_target_channels, device):

        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.to(device)
        self.vgg_style_layers = [1, 6, 11, 18, 25]
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None].to(device)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225])[:,None,None].to(device)

        self.target_style_image = target_style_image
        self.num_target_channels = num_target_channels
        self.device = device

        with torch.no_grad():
            self.target_vgg_features = self.calc_styles_vgg(self.to_nchw(target_style_image).to(device))

    def calc_styles_vgg(self, imgs):

        imgs = imgs.to(self.device)
        x = (imgs-self.vgg_mean) / self.vgg_std
        b, c, h, w = x.shape
        features = [x.reshape(b, c, h*w)]
        for i, layer in enumerate(self.vgg16[:max(self.vgg_style_layers)+1]):
            x = layer(x)
            if i in self.vgg_style_layers:
                b, c, h, w = x.shape
                features.append(x.reshape(b, c, h*w))
        return features

    def project_sort(self, x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    def ot_loss(self, source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n), dim=0).to(self.device)
        source_proj = self.project_sort(source, projs)
        target_proj = self.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return torch.log10((source_proj-target_interp).square().sum())

    def calc_loss(self, preds):
        with torch.no_grad():
            xx = self.calc_styles_vgg(preds[:, : self.num_target_channels, :, :])
        return sum(self.ot_loss(x, y) for x, y in zip(xx, self.target_vgg_features))

    def to_nchw(self, img):
        transform = transforms.ToTensor()
        return transform(img).unsqueeze(0)
