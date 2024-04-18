from datetime import datetime

import torch

from utils.dataset import ConditioningDataset
from nca import ConditionedNCA
from conditioned_trainer import ConditionedNCATrainer
from utils.utils import load_target_style_image

if __name__ == "__main__":

    NUM_HIDDEN_CHANNELS = 16

    dataset = ConditioningDataset('../../../data/random_faces/', image_size=64)
    target_style_image = load_target_style_image('../../../data/style_images/picasso.jpg', size=64)

    nca = ConditionedNCA(
            target_shape = dataset.target_size,
            num_hidden_channels = NUM_HIDDEN_CHANNELS,
            living_channel_dim = 3,
            cell_fire_rate = 0.5
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
            lr = 1e-3,
            pool_size = 1024,
            num_damaged = 0,
            log_base_path = "tensorboard_logs",
            damage_radius = 3,
            appearance_loss_type = 'OT',
            appearance_loss_weight = 1.0,
            content_loss_weight = .1 ,
            overflow_loss_weight = 1.0,
            device = device,
        )

    print(nca)

    try:
        trainer.train(batch_size=8, epochs=100000)
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError) as e:
        print(e)
        nca.save("models/ConditionedNCA_{}.pt".format(datetime.now()))
