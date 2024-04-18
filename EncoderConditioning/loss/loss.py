'''
Modified from base implementation in:
    Pajouheshgar, E., Xu, Y., Zhang, T., & Süsstrunk, S. (2023).
    "Dynca: Real-time dynamic texture synthesis using neural cellular automata"
     In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

https://github.com/IVRL/DyNCA/blob/main/utils/loss/loss.py
'''

import torch
import numpy as np

from .appearance_loss import AppearanceLoss
from .content_loss import ContentLoss


class Loss(torch.nn.Module):
    def __init__(self, device, content_loss_weight=1.0, overflow_loss_weight=1.0,
                 appearance_loss_weight=1.0, appearance_loss_type='OT',
                 target_style_image=None):
        super(Loss, self).__init__()

        self.device = device

        self.appearance_loss_type = appearance_loss_type
        self.appearance_loss_weight = appearance_loss_weight
        self.target_style_image = target_style_image

        self.content_loss_weight = content_loss_weight
        self.overflow_loss_weight = overflow_loss_weight

        if self.appearance_loss_weight > 0:
            assert target_style_image is not None, 'Target style image required to use appearance loss'

        self._create_losses()

    def get_overflow_loss(self, input_dict):
        nca_state = input_dict['nca_state']
        overflow_loss = (nca_state - nca_state.clamp(-1.0, 1.0)).abs().mean()
        return overflow_loss


    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.overflow_loss_weight != 0:
            self.loss_mapper["overflow"] = self.get_overflow_loss
            self.loss_weights["overflow"] = self.overflow_loss_weight

        if self.appearance_loss_weight != 0:
            self.loss_mapper["appearance"] = AppearanceLoss(self.target_style_image,
                                                            self.device,
                                                            self.appearance_loss_type)
            self.loss_weights["appearance"] = self.appearance_loss_weight

        if self.content_loss_weight != 0:
            self.loss_mapper["content"] = ContentLoss(self.device)
            self.loss_weights["content"] = self.content_loss_weight

    def forward(self, input_dict, return_summary=True):
        loss = 0
        loss_log_dict = {}
        for loss_name in self.loss_mapper:
            l = self.loss_mapper[loss_name](input_dict)

            l *= self.loss_weights[loss_name]
            loss_log_dict[loss_name] = l.item()
            loss += l

        output = [loss]
        if return_summary:
            output.append(loss_log_dict)
        else:
            output.append(None)
        return output
