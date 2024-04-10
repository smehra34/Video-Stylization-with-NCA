import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from einops import rearrange
import os

from .utils import load_image

class ConditioningDataset(Dataset):

    def __init__(self, data_dir, image_size=64):

        # get a list of paths to the images in the data directory
        images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if os.path.splitext(f)[-1] in ['.jpg', '.png']]
        # load all images
        self.images = torch.stack([load_image(im_path, image_size) for im_path in images], dim=0)

        self.num_samples = self.images.size(0)
        self.targets = torch.arange(self.num_samples)
        self.target_size = self.images.size()[-3:]

    def num_goals(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.images[idx : idx + 1].clone(), idx
        return self.images[idx].clone(), idx

    def __len__(self):
        return self.num_samples

    def to(self, device: torch.device):
        self.images = self.images.to(device)

    def visualize(self, idx=0):
        self.plot_img(self.images[idx : idx + 1])

    def plot_img(self, img):
        with torch.no_grad():
            img = img.squeeze().detach().cpu().numpy()
        img = rearrange(img, "c w h -> w h c")
        plt.imshow(img)
        plt.show()
