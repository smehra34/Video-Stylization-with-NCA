'''
Modified from base implementation in:
    Pajouheshgar, E., Xu, Y., Zhang, T., & Süsstrunk, S. (2023).
    "Dynca: Real-time dynamic texture synthesis using neural cellular automata"
     In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

https://github.com/IVRL/DyNCA/blob/main/utils/loss/appearance_loss.py
'''

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms

import torchvision.models as torch_models
import numpy as np


class AppearanceLoss(torch.nn.Module):
    def __init__(self, target_style_image, device, appearance_loss_type='OT'):
        super(AppearanceLoss, self).__init__()

        self.device = device

        self.target_style_image = target_style_image
        self.target_style_tensor = self._to_nchw(target_style_image).to(device)

        self.appearance_loss_type = appearance_loss_type

        self.slw_weight = 0.0
        self.ot_weight = 0.0
        self.gram_weight = 0.0
        if self.appearance_loss_type == 'OT':
            self.ot_weight = 1.0
        elif self.appearance_loss_type == 'SlW':
            self.slw_weight = 1.0
        elif self.appearance_loss_type == 'Gram':
            self.gram_weight = 1.0

        self._create_losses()

    def _to_nchw(self, img):
        transform = transforms.ToTensor()
        return transform(img).unsqueeze(0)

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}
        if self.slw_weight != 0:
            self.loss_mapper["SlW"] = SlicedWassersteinLoss(self.target_style_tensor, self.device)
            self.loss_weights["SlW"] = self.slw_weight

        if self.ot_weight != 0:
            self.loss_mapper["OT"] = OptimalTransportLoss(self.target_style_tensor, self.device)
            self.loss_weights["OT"] = self.ot_weight

        if self.gram_weight != 0:
            self.loss_mapper["Gram"] = GramLoss(self.target_style_tensor, self.device)
            self.loss_weights["Gram"] = self.gram_weight

    def update_losses_to_apply(self, epoch):
        pass

    def forward(self, input_dict):

        loss = 0.0
        generated_images = input_dict['generated_images']

        # # Scale the images before feeding to VGG
        # generated_images = (generated_images + 1.0) / 2.0
        # target_images = (target_images + 1.0) / 2.0

        for loss_name in self.loss_mapper:
            loss_weight = self.loss_weights[loss_name]
            loss_func = self.loss_mapper[loss_name]
            loss += loss_weight * loss_func(generated_images)
        return loss


class GramLoss(torch.nn.Module):
    def __init__(self, target_style_tensor, device):
        super(GramLoss, self).__init__()

        self.device = device
        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(device)

        with torch.no_grad():
            self.target_features = get_middle_feature_vgg(self.device, target_style_tensor, self.vgg16)

    @staticmethod
    def get_gram(y):
        b, c, h, w = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        grams = features.bmm(features_t) / (h * w)
        return grams

    def forward(self, generated_images):
        generated_features = get_middle_feature_vgg(self.device, generated_images, self.vgg16)

        loss = 0.0
        for target_feature, generated_feature in zip(self.target_features, generated_features):
            gram_target = self.get_gram(target_feature)
            gram_generated = self.get_gram(generated_feature)
            loss = loss + (gram_target - gram_generated).square().mean()
        return loss


class SlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, target_style_tensor, device):
        super(SlicedWassersteinLoss, self).__init__()

        self.device = device
        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(device)

        with torch.no_grad():
            self.target_features = get_middle_feature_vgg(self.device, target_style_tensor, self.vgg16,
                                                          flatten=True, include_image_as_feat=True)

    @staticmethod
    def project_sort(x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    def sliced_ot_loss(self, source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n), dim=0).to(self.device)
        source_proj = SlicedWassersteinLoss.project_sort(source, projs)
        target_proj = SlicedWassersteinLoss.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return (source_proj - target_interp).square().sum()

    def forward(self, generated_images):
        generated_features = get_middle_feature_vgg(self.device, generated_images, self.vgg16, flatten=True,
                                                    include_image_as_feat=True)

        return sum(self.sliced_ot_loss(x, y) for x, y in zip(generated_features, self.target_features))


class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, target_style_tensor, device):
        super(OptimalTransportLoss, self).__init__()

        self.device = device
        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(device)

        with torch.no_grad():
            self.target_features = get_middle_feature_vgg(self.device, target_style_tensor, self.vgg16)

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
        dist = 1. - torch.mm(x, y_t) / (x_norm + 1e-10) / (y_norm + 1e-10)
        return dist

    @staticmethod
    def style_loss_cos(X, Y, cos_d=True):
        # X,Y: 1*d*N*1
        d = X.shape[1]

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)  # N*d
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = OptimalTransportLoss.pairwise_distances_cos(X, Y)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())

        return remd

    @staticmethod
    def moment_loss(X, Y):  # matching mean and cov
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = mu_d + D_cov

        return loss

    @staticmethod
    def get_ot_loss_single_batch(x_feature, y_feature):
        randomize = True
        loss = 0
        for x, y in zip(x_feature, y_feature):
            c = x.shape[1]
            h, w = x.shape[2], x.shape[3]
            x = x.reshape(1, c, -1, 1)
            y = y.reshape(1, c, -1, 1)
            if h > 32 and randomize:
                indices = np.random.choice(np.arange(h * w), size=1000, replace=False)
                indices = np.sort(indices)
                indices = torch.LongTensor(indices)
                x = x[:, :, indices, :]
                y = y[:, :, indices, :]
            loss += OptimalTransportLoss.style_loss_cos(x, y)
            loss += OptimalTransportLoss.moment_loss(x, y)
        return loss

    def forward(self, generated_images):
        generated_features = get_middle_feature_vgg(self.device, generated_images, self.vgg16)
        batch_size = len(generated_images)
        loss = 0.0
        for b in range(batch_size):
            generated_feature = [g[b:b + 1] for g in generated_features]
            loss += self.get_ot_loss_single_batch(self.target_features, generated_feature)
        return loss / batch_size


def get_middle_feature_vgg(device, imgs, vgg_model, flatten=False, include_image_as_feat=False):

    img_shape = imgs.shape[2]
    style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(device)

    x = (imgs - mean) / std
    b, c, h, w = x.shape
    if (include_image_as_feat):
        features = [x.reshape(b, c, h * w)]
    else:
        features = []
    for i, layer in enumerate(vgg_model[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            if flatten:
                features.append(x.reshape(b, c, h * w))
            else:
                features.append(x)
    return features
