from datetime import datetime

import torch

from utils.dataset import ConditioningDataset
from nca import ConditionedNCA
from conditioned_trainer import ConditionedNCATrainer
from utils.utils import load_target_style_image

import argparse
import os
import shutil
import json

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='Train conditional NCA')

    ap.add_argument("-d", "--conditioning-dataset",
                    help=("Path to the directory containing the dataset to " +
                          "condition on during training. "))

    ap.add_argument("-t", "--target-style-image",
                    help="Path to the target style image.")

    ap.add_argument("-n", "--exp-name", default='test',
                    help="Name for the experiment.")

    ap.add_argument('--overwrite', action='store_true',
                    help="Overwrite previous experiment with same name (if exists).")

    ap.add_argument("-O", "--log-dir", default='logs',
                    help=("Path to directory for storing experiment logs." +
                          "Includes tensorboard logs and model checkpoints. Default 'logs'."))

    ap.add_argument("-N", "--num-hidden-channels", type=int, default=16,
                    help=("Num hidden channels in NCA. Default 16"))

    ap.add_argument("-s", "--img-size", type=int, default=64,
                    help=("Dimension of images for training. Default 64."))

    ap.add_argument("-b", "--batch-size", type=int, default=8,
                    help=("Batch size. Default 8."))

    ap.add_argument("-l", "--learning-rate", type=float, default=1e-3,
                    help=("Learning rate. Default 1e-3."))

    ap.add_argument("-e", "--epochs", type=int, default=100000,
                    help=("Number of epochs. Default 100,000."))

    ap.add_argument("-f", "--cell-fire-rate", type=float, default=0.5,
                    help=("Cell fire rate. Default 0.5."))

    ap.add_argument("-p", "--pool-size", type=int, default=1024,
                    help=("Pool size. Default 1024."))

    ap.add_argument("-r", "--damage-radius", type=int, default=3,
                    help=("Damage radius. Default 3."))

    ap.add_argument("-A", "--appearance-loss-type", default='OT',
                    help=("Appearance loss type. One of 'OT', 'SlW' or 'Gram'." +
                          "Default 'OT'."))

    ap.add_argument("-a", "--appearance-loss-weight", type=float, default=1.0,
                    help=("Appearance loss weight. Default 1.0."))

    ap.add_argument("-c", "--content-loss-weight", type=float, default=0.1,
                    help=("Content loss weight. Default 0.1."))

    ap.add_argument("-o", "--overflow-loss-weight", type=float, default=1.0,
                    help=("Overflow loss weight. Default 1.0."))

    args = ap.parse_args()


    outdir = os.path.join(args.log_dir, args.exp_name)

    if os.path.exists(outdir):
        assert args.overwrite, (f"Results for experiment '{args.exp_name}' " +
                                "already exist. Please specify the " +
                                "--overwrite option if you wish to " +
                                "overwrite these results.")
        print("Overwriting previous results")
        shutil.rmtree(outdir)

    os.makedirs(outdir)
    print(f"Created results directory: {outdir}")

    args_log_file = os.path.join(outdir, 'args.json')
    with open(args_log_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    print(f"Saved arguments log file to {args_log_file}")

    NUM_HIDDEN_CHANNELS = args.num_hidden_channels

    dataset = ConditioningDataset(args.conditioning_dataset, image_size=args.img_size)
    target_style_image = load_target_style_image(args.target_style_image, size=args.img_size)

    nca = ConditionedNCA(
            target_shape = dataset.target_size,
            num_hidden_channels = NUM_HIDDEN_CHANNELS,
            living_channel_dim = 3,
            cell_fire_rate = args.cell_fire_rate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running training on", device)

    nca = nca.to(device)
    dataset.to(device)

    trainer = ConditionedNCATrainer(
            nca,
            dataset,
            target_style_image,
            nca_steps=[48, 96],
            lr = args.learning_rate,
            pool_size = args.pool_size,
            num_damaged = 0,
            log_base_path = outdir,
            damage_radius = args.damage_radius,
            appearance_loss_type = args.appearance_loss_type,
            appearance_loss_weight = args.appearance_loss_weight,
            content_loss_weight = args.content_loss_weight,
            overflow_loss_weight = args.overflow_loss_weight,
            device = device,
        )

    print(nca)

    try:
        trainer.train(batch_size=args.batch_size, epochs=args.epochs)
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError) as e:
        print(e)
        print('Saving latest model checkpoint...')

    nca.save(f"{outdir}/ConditionedNCA.pt")
