import os
import warnings

import wandb
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy

from collections import defaultdict

from models.dynca import DyNCA

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

from utils.misc.display_utils import save_train_image
from utils.misc.preprocess_texture import preprocess_style_image, preprocess_target_images, RGBToGrayscale
from utils.misc.video_utils import VideoWriter
import matplotlib.pyplot as plt
from utils.misc.flow_viz import plot_vec_field

from utils.loss.loss import Loss
from IPython.display import clear_output, Markdown

import argparse
from helper import *
from utils.misc.video_utils import evaluate_folder_of_videos, generate_control_videos, save_video

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# torch.backends.cudnn.deterministic = True

def setup_args():
    parser = argparse.ArgumentParser(description='Experimental setup for DyNCA')

    # General settings
    parser.add_argument('--exp_name', type=str, default='no-positional-encoding-with-motion-loss', help='Name of the experiment')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256], help='Image size during training')
    parser.add_argument('--style_name', type=str, default='starry-night', help='Name of the style image')

    parser.add_argument('--exp_description', type=str, default='', help='Description of the experiment')

    # NCA related settings

    parser.add_argument('--nca_pool_size', type=int, default=256, help='Pool size for NCA')
    parser.add_argument('--nca_step_range', type=int, nargs=2, default=[32, 128], help='Step range for NCA')
    parser.add_argument('--nca_inject_seed_step', type=int, default=8, help='Injection step for NCA seed')
    parser.add_argument('--nca_c_in', type=int, default=12, help='Input channel of DyNCA')
    parser.add_argument('--nca_fc_dim', type=int, default=96, help='Dimensionality of update MLP in DyNCA')
    parser.add_argument('--nca_seed_mode', type=str, default='zeros', help='Seed initialization mode')
    parser.add_argument('--nca_padding_mode', type=str, default='circular', help='Padding mode in perception')
    parser.add_argument('--nca_conditioning', type=str, default='edges', help='Conditioning type. Either pos_emb, edges or None. Default edges.')
    parser.add_argument('--edge_transform', type=str, default='None', help='Transformation to apply to the edges. Either tanh or None. Default None.')
    parser.add_argument('--nca_perception_scales', type=int, nargs='+', default=[0], help='Perception scales for NCA')
    parser.add_argument('--nca_base_num_steps', type=int, default=24, help='Base number of steps for NCA')


    # motion loss parameters

    parser.add_argument('--motion_img_size', type=int, nargs=2, default=[256, 256], help='Image size for motion field')
    # motion_vector_field_name: 'circular', 'random', 'horizontal', 'vertical', 'diagonal'
    parser.add_argument('--motion_vector_field_name', type=str, default='circular', help='Name of the motion vector field')
    # motion_model_name: 'two_stream_dynamic', 'two_stream_static', 'one_stream'
    parser.add_argument('--motion_model_name', type=str, default='two_stream_dynamic', help='Name of the motion model')
    parser.add_argument('--motion_strength_weight', type=float, default=0.5, help='Weight for motion strength in loss')
    parser.add_argument('--motion_direction_weight', type=float, default=0.5, help='Weight for motion direction in loss')
    parser.add_argument('--motion_weight_change_interval', type=int, default=500, help='Interval for changing motion weight')
    parser.add_argument('--vector_field_motion_loss_weight', type=float, default=1.0, help='Weight for vector field motion loss')

    # Training settings
    parser.add_argument('--max_iterations', type=int, default=2000, help='Maximum number of iterations')
    parser.add_argument('--save_every', type=int, default=50, help='Save frequency')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, nargs='+', action='append', default=[[500, 1000]], help='Steps for learning rate decay')
    parser.add_argument('--DEVICE', type=str, default='cuda', help='Device to run the training on')


    # Video generation params
    parser.add_argument('--video_size_factor', type=float, default=2.0, help='Size factor for video generation')
    parser.add_argument('--video_steps_per_frame', type=int, default=1, help='Number of steps per frame in video generation')

    return parser.parse_args()

def main():
    args = setup_args()

    DEVICE = torch.device(args.DEVICE)

    exp_dir = f'experiments/experiment_{args.exp_description}'
    model_save_path = f'{exp_dir}/models/'
    ensure_dir(model_save_path)

    # Load the style image
    style_images_path = 'data/Style_images/'
    ensure_dir(style_images_path)
    style_img_path = find_image_by_name(style_images_path, args.style_name)
    style_img = Image.open(style_img_path)
    # Preprocess the style image
    target_appearance_img = preprocess_style_image(style_img, model_type='vgg',
                                               img_size=args.img_size,
                                               batch_size=args.batch_size) * 2.0 - 1.0  # [-1.0, 1.0]



    ###### setup the DyNCA model for training ######
    nca_size_x, nca_size_y = int(args.img_size[0]), int(args.img_size[1])

    try:
        nca_perception_scales = args.nca_perception_scales
    except:
        nca_perception_scales = [0]
    assert nca_perception_scales[0] == 0

    nca_min_steps, nca_max_steps = args.nca_step_range

    nca_model = DyNCA(c_in=args.nca_c_in, c_out=3, fc_dim=args.nca_fc_dim,
                    seed_mode=args.nca_seed_mode,
                    conditioning=args.nca_conditioning,
                    edge_transform=args.edge_transform,
                    padding_mode=args.nca_padding_mode,
                    perception_scales=nca_perception_scales,
                    device=DEVICE)


    with torch.no_grad():
        nca_pool = nca_model.seed(args.nca_pool_size, size=(nca_size_x, nca_size_y))


    nca_model.load_state_dict(torch.load(model_save_path + f'model_checkpoint.pth'))
    print('Model loaded from:', model_save_path + f'model_checkpoint.pth')
    param_n = sum(p.numel() for p in nca_model.parameters())
    print('DyNCA param count:', param_n)

    video_save_path = f'{exp_dir}/videos/'
    ensure_dir(video_save_path)
    ## Generate Videos
    generate_control_videos(style_img_path, video_save_path, size_factor=args.video_size_factor,
                              step_n=int(args.nca_base_num_steps), steps_per_frame=args.video_steps_per_frame,
                              nca_model=nca_model, nca_size_x=nca_size_x,
                              nca_size_y=nca_size_y, DEVICE=DEVICE)
    evaluate_folder_of_videos('data/Evaluation', video_save_path, size_factor=args.video_size_factor,
                              step_n=int(args.nca_base_num_steps), steps_per_frame=args.video_steps_per_frame,
                              nca_model=nca_model, nca_size_x=nca_size_x,
                              nca_size_y=nca_size_y, DEVICE=DEVICE)

if __name__ == '__main__':
    main()
