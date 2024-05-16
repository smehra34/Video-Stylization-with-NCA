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
from utils.misc.preprocess_texture import preprocess_style_image, preprocess_video, RGBToEdges
from utils.misc.video_utils import VideoWriter
import matplotlib.pyplot as plt
from utils.misc.flow_viz import plot_vec_field

from utils.loss.loss import Loss
from IPython.display import clear_output, Markdown

import argparse
from helper import *
from utils.misc.video_utils import save_video

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# torch.backends.cudnn.deterministic = True

def setup_args():
    parser = argparse.ArgumentParser(description='Experimental setup for DyNCA')

    # General settings
    parser.add_argument('--exp_name', type=str, default='no-positional-encoding-with-motion-loss', help='Name of the experiment')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256], help='Image size during training')
    parser.add_argument('--video_length', type=int, default=10, help='Output video length in seconds')
    parser.add_argument('--style_name', type=str, default='starry-night', help='Name of the style image')
    parser.add_argument('--style_img_ext', type=str, default='jpg', help='Extension of the style image')
    parser.add_argument('--target_appearance_2_path', type=str, default='data/Reference/mr-bean.gif', help='Path to the second target appearance reference')


    # NCA related settings

    parser.add_argument('--nca_pool_size', type=int, default=256, help='Pool size for NCA')
    parser.add_argument('--nca_step_range', type=int, nargs=2, default=[32, 128], help='Step range for NCA')
    parser.add_argument('--nca_inject_seed_step', type=int, default=8, help='Injection step for NCA seed')
    parser.add_argument('--nca_c_in', type=int, default=12, help='Input channel of DyNCA')
    parser.add_argument('--nca_fc_dim', type=int, default=96, help='Dimensionality of update MLP in DyNCA')
    parser.add_argument('--nca_seed_mode', type=str, default='zeros', help='Seed initialization mode')
    parser.add_argument('--nca_padding_mode', type=str, default='circular', help='Padding mode in perception')
    parser.add_argument('--nca_pos_emb', type=str, default=None, help='Positional encoding type')
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


    # appearance loss parameters

    parser.add_argument('--appearance_loss_weight', type=float, default=1.0, help='Weight for appearance loss')
    # appearance_loss_type: 'OT', 'MSE'
    parser.add_argument('--appearance_loss_type', type=str, default='OT', help='Type of appearance loss')


    # auxiliary loss parameters

    parser.add_argument('--auxillary_loss_weight', type=float, default=10.0, help='Weight for auxiliary loss')
    # auxillary_loss_type: 'MSE', 'OT'
    parser.add_argument('--auxillary_loss_type', type=str, default='VGG', help='Type of auxiliary loss')

    # overflow loss parameters
    parser.add_argument('--overflow_loss_weight', type=float, default=1000.0, help='Weight for overflow loss')

    # Training settings
    parser.add_argument('--max_iterations', type=int, default=2000, help='Maximum number of iterations')
    parser.add_argument('--save_every', type=int, default=25, help='Save frequency')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, nargs='+', action='append', default=[[500, 1000]], help='Steps for learning rate decay')
    parser.add_argument('--DEVICE', type=str, default='cuda', help='Device to run the training on')

    return parser.parse_args()

def main():
    args = setup_args()
    experiment_index = get_next_experiment_index('experiments')
    args.exp_description = f"{args.exp_name}_{experiment_index}"

    wandb.init(
        # set the wandb project where this run will be logged
        project='Condiitional-NCA',
        entity='video_stylization_with_NCAs',
        config = {'experiment_index': experiment_index, **vars(args)},
        name = args.exp_description
        )

    DEVICE = torch.device(args.DEVICE)

    # Load the style image
    style_img= Image.open(f"data/VectorFieldMotion/Appearance/{args.style_name}.{args.style_img_ext}")
    # Preprocess the style image
    target_appearance_img = preprocess_style_image(style_img, model_type='vgg',
                                               img_size=args.img_size,
                                               batch_size=args.batch_size) * 2.0 - 1.0  # [-1.0, 1.0]
    # Load the target appearance reference
    target_reference_img = preprocess_video(args.target_appearance_2_path,
                                               img_size=args.img_size)  # [C, T, H, W]
    target_appearance_img = target_appearance_img.to(DEVICE)
    target_reference_img = target_reference_img.permute(1,0,2,3).to(DEVICE)

    # convert the target appearance to edges and save the setup images
    edge_converter = RGBToEdges().to(DEVICE)
    target_reference_edges = edge_converter(target_reference_img)
    save_setup_images(target_reference_img, target_reference_edges, target_appearance_img)


    ###### setup the DyNCA model for training ######
    nca_size_x, nca_size_y = int(args.img_size[0]), int(args.img_size[1])

    try:
        nca_perception_scales = args.nca_perception_scales
    except:
        nca_perception_scales = [0]
    assert nca_perception_scales[0] == 0

    nca_min_steps, nca_max_steps = args.nca_step_range

    nca_model = DyNCA(c_in=args.nca_c_in+3, c_out=3, fc_dim=args.nca_fc_dim,
                    seed_mode=args.nca_seed_mode,
                    pos_emb=args.nca_pos_emb, padding_mode=args.nca_padding_mode,
                    perception_scales=nca_perception_scales,
                    device=DEVICE)
    with torch.no_grad():
        nca_pool = nca_model.seed(args.nca_pool_size, size=(nca_size_x, nca_size_y))

    param_n = sum(p.numel() for p in nca_model.parameters())
    print('DyNCA param count:', param_n)

    optimizer = torch.optim.Adam(nca_model.parameters(), lr=args.lr)

    DynamicTextureLoss = Loss(args, nca_model)

    args_log = copy.deepcopy(args.__dict__)
    del args_log['DEVICE']
    if 'target_motion_vec' in args_log:
        del args_log['target_motion_vec']


    if len(args.lr_decay_step) == 0:
        args.lr_decay_step = [[1000, 2000]]

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        args.lr_decay_step[0],
                                                        0.5)

    input_dict = {}  # input dictionary for computing the loss functions
    input_dict['target_image_list'] = [target_appearance_img]  # 0,1
    input_dict['target_image_edges'] = edge_converter(target_appearance_img)[0]

    interval = args.motion_weight_change_interval

    loss_log_dict = defaultdict(list)
    new_size = (nca_size_x, nca_size_y)

    pbar = range(args.max_iterations)

    ###### Training loop ######
    try:
        for i in pbar:
            print(i)
            np.random.seed(i + 424)
            torch.manual_seed(i + 424)
            torch.cuda.manual_seed_all(i + 424)

            aux_img_ids = np.random.choice(target_reference_img.size(0), args.batch_size, replace=True)
            aux_imgs = [target_reference_img[img].unsqueeze(0) for img in aux_img_ids]
            aux_imgs = torch.concat(aux_imgs)
            aux_imgs_vis = (aux_imgs + 1.0) / 2.0
            input_dict['auxillary_image_list'] = [aux_imgs]  # 0,1

            aux_edges = [target_reference_edges[img].unsqueeze(0) for img in aux_img_ids]
            aux_edges = torch.concat(aux_edges)
            aux_edges_vis = (aux_edges + 1.0) / 2.0

            with torch.no_grad():
                batch_idx = np.random.choice(args.nca_pool_size, args.batch_size, replace=False)
                input_states = nca_pool[batch_idx]
                seed_injection = False
                if i % args.nca_inject_seed_step == 0:
                    seed_injection = True
                    seed_inject = nca_model.seed(1, size=(nca_size_x, nca_size_y))
                    input_states[:1] = seed_inject[:1]

                input_states = torch.cat((input_states, aux_edges), 1)
                '''Get the image before NCA iteration for computing optic flow'''
                nca_states_before, nca_features_before = nca_model.forward_nsteps(input_states, step_n=1)
                z_before_nca = nca_features_before
                image_before_nca = z_before_nca

            step_n = np.random.randint(nca_min_steps, nca_max_steps)
            input_dict['step_n'] = step_n
            nca_states_after, nca_features_after = nca_model.forward_nsteps(input_states, step_n)

            z = nca_features_after
            generated_image = z
            with torch.no_grad():
                generated_image_vis = generated_image.clone()
                generated_image_vis = (generated_image_vis + 1.0) / 2.0

            image_after_nca = generated_image.clone()

            '''Construct input dictionary for loss computation'''
            input_dict['generated_image_list'] = [generated_image]
            input_dict['generated_image_before_nca'] = image_before_nca
            input_dict['generated_image_after_nca'] = image_after_nca



            input_dict['nca_state'] = nca_states_after

            batch_loss, batch_loss_log_dict, summary = DynamicTextureLoss(input_dict, return_summary=True)

            #if i % args.save_every == 0:
        #        batch_loss, batch_loss_log_dict, summary = DynamicTextureLoss(input_dict, return_summary=True)
    #            print('batch_loss', batch_loss)
#                print('batch_loss_log_dict', batch_loss_log_dict)
#                print('summary', summary)

            with torch.no_grad():
                batch_loss.backward()
                if torch.isnan(batch_loss):
                    print('Loss is NaN. Train Failed. Exit.')
                    exit()

                for p_name, p in nca_model.named_parameters():
                    p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                nca_pool[batch_idx] = nca_states_after[:, :12, :, :]

                display_dict = copy.deepcopy(batch_loss_log_dict)
                    # Assuming 'batch_loss', 'batch_loss_log_dict', and 'summary' are obtained during your training loop
                wandb.log({
                    "batch_loss": batch_loss.item(),  # Log the scalar value of batch loss
                    **display_dict,            # Expand the dictionary to log each component separately
                })

                # Log the generated and reference images
                img_show = save_train_image(generated_image_vis.detach().cpu().numpy(), None, return_img = True)
                ref_img_show = save_train_image(aux_imgs_vis.detach().cpu().numpy(), None, return_img = True)
                wandb.log({"Generated Image": wandb.Image(img_show)})
                wandb.log({"Reference Image": wandb.Image(ref_img_show)})

                # Additionally, to log images stored in 'summary':
                if 'vector_field_motion-generated_video_flow' in summary:
                    # Extract the numpy array containing video flow
                    video_flow_data = summary['vector_field_motion-generated_video_flow']

                    # Assuming the data shape is (num_images, channels, height, width)
                    # We'll log the first image in the batch for simplicity
                    if video_flow_data.ndim == 4 and video_flow_data.shape[1] == 3:  # Check if it is in the format (N, C, H, W)
                        # Convert from (N, C, H, W) to (N, H, W, C) for the first image
                        first_image = video_flow_data[0].transpose(1, 2, 0)  # This converts C, H, W to H, W, C
                        video_flow_img = wandb.Image(first_image)  # Create a wandb Image from numpy array
                        wandb.log({"Generated Video Flow": video_flow_img})  # Log the image to wandb

                # Log generated and target flow vector fields
                if 'vector_field_motion-generated_flow_vector_field' in summary:
                    # Directly log the PIL image, assuming it is correctly formatted
                    generated_flow_vector_field_img = wandb.Image(summary['vector_field_motion-generated_flow_vector_field'])
                    wandb.log({"Generated Flow Vector Field": generated_flow_vector_field_img})

                if 'vector_field_motion-target_flow_vector_field' in summary:
                    # Directly log the PIL image, assuming it is correctly formatted
                    target_flow_vector_field_img = wandb.Image(summary['vector_field_motion-target_flow_vector_field'])
                    wandb.log({"Target Flow Vector Field": target_flow_vector_field_img})

                #else:
            #        batch_loss, batch_loss_log_dict, _ = DynamicTextureLoss(input_dict, return_summary=False)
            #        summary = {}


    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError) as e:
        print(e)
        print('Saving latest model checkpoint...')
    model_save_path = f'experiments/experiment_{experiment_index}/models/'
    ensure_dir(model_save_path)
    torch.save(nca_model, model_save_path + f'model_{i}.pth')

    video_save_path = f'experiments/experiment_{experiment_index}/videos/'
    ensure_dir(video_save_path)
    ## Generate Videos
    ## CORGI
    save_video(f"{video_save_path}/corgi", target_vid_path='data/Reference/corgi.gif', size_factor=2.0, step_n=int(args.nca_base_num_steps), steps_per_frame=10, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
    ##GOT
    save_video(f"{video_save_path}/got", target_vid_path='data/Reference/got.gif', size_factor=3.0, step_n=int(args.nca_base_num_steps), steps_per_frame=2, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
    ##MR BEAN
    save_video(f"{video_save_path}/mr_bean", target_vid_path='data/Reference/mr-bean.gif', size_factor=2.0, step_n=int(args.nca_base_num_steps), steps_per_frame=2, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
    ##STARRY NIGHT
    save_video(f"{video_save_path}/{args.style_name}", target_vid_path=f'data/Reference/{args.style_name}.gif', size_factor=2.0, step_n=int(args.nca_base_num_steps), steps_per_frame=2, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
    # Raed Webcam
    save_video(f"{video_save_path}/raed_webcam", target_vid_path='data/Reference/raed.mp4', size_factor=2.0, step_n=int(args.nca_base_num_steps), steps_per_frame=2, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
    # Black background
    save_video(f"{video_save_path}/black_background", target_vid_path='data/Reference/black_background.mp4', size_factor=2.0, step_n=int(args.nca_base_num_steps), steps_per_frame=2, is_style_image=False, nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)
if __name__ == '__main__':
    main()
