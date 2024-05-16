import os
import torch
from PIL import Image
import numpy as np

from utils.misc.display_utils import save_train_image
from utils.misc.preprocess_texture import preprocess_style_image, preprocess_video, RGBToEdges
import matplotlib.pyplot as plt
from utils.misc.flow_viz import plot_vec_field

from utils.loss.loss import Loss
from IPython.display import clear_output, Markdown
from tqdm import tqdm

def ensure_dir(directory):
    """ Ensure the directory exists, and if not, create it. """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_experiment_index(base_directory):
    """ Determine the next experiment index based on existing directories. """
    if not os.path.exists(base_directory):
        return 1  # Start with 1 if the base directory does not exist
    existing_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    existing_indexes = [int(d.split('_')[-1]) for d in existing_dirs if d.startswith('experiment_') and d.split('_')[-1].isdigit()]
    if existing_indexes:
        return max(existing_indexes) + 1
    return 1


def save_setup_images(target_reference_img, target_reference_edges, target_appearance_img):
    """ Save the provided images into an experiment-specific subfolder named 'setup_images'. """
    base_directory = 'experiments'
    experiment_index = get_next_experiment_index(base_directory)
    directory = os.path.join(base_directory, f'experiment_{experiment_index}', 'setup_images')
    ensure_dir(directory)

    # Process and save the target reference image
    # Normalize from [-1, 1] to [0, 255] for color images
    img = (target_reference_img[0].detach().cpu().permute(1, 2, 0).numpy() + 1) * 127.5
    Image.fromarray(img.astype(np.uint8)).save(os.path.join(directory, 'target_reference_image.png'))

    # Process and save the target reference edges

    edges = target_reference_edges[0][1].detach().cpu().numpy()
    edges_normalized = (edges - edges.min()) / (edges.max() - edges.min()) * 255
    Image.fromarray(edges_normalized.astype(np.uint8), mode='L').save(os.path.join(directory, 'target_reference_edges.png'))

    # Process and save the target appearance image
    # Normalize from [-1, 1] to [0, 255] for color images
    appearance = (target_appearance_img[0].detach().cpu().permute(1, 2, 0).numpy() + 1) * 127.5
    Image.fromarray(appearance.astype(np.uint8)).save(os.path.join(directory, 'target_appearance_image.png'))
