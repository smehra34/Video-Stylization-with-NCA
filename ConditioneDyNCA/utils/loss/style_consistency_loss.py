import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class StyleConsistencyLoss(torch.nn.Module):
    def __init__(self, args, nca_model):
        super(StyleConsistencyLoss, self).__init__()
        self.args = args
        self.nca_model = nca_model
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, input_dict, return_summary=True):
        
        with torch.no_grad():
    
            target_image = input_dict['target_image_list'][0]
            target_image_edges = input_dict['target_image_edges'].unsqueeze(0)
            nca_size_x, nca_size_y = int(self.args.img_size[0]), int(self.args.img_size[1])
            h = self.nca_model.seed(1, size=(nca_size_x, nca_size_y))
                            
            h = torch.cat((h, target_image_edges), 1)
    
            # grow for min steps
            step_n = self.args.nca_step_range[0]
            nca_state, nca_feature = self.nca_model.forward_nsteps(h, step_n)
    
            z = nca_feature    
            #img = z.detach().cpu().numpy()[0]
            #img = img.transpose(1, 2, 0)
    
            #img = np.clip(img, -1.0, 1.0)
            #img = (img + 1.0) / 2.0
    
            return self.loss_fn(z, target_image), None, None
