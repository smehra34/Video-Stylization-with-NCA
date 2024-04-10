from datetime import datetime

import torch

from utils.dataset import ConditioningDataset
from nca import ConditionedNCA
from conditioned_trainer import ConditionedNCATrainer


if __name__ == "__main__":

    NUM_HIDDEN_CHANNELS = 16

    dataset = ConditioningDataset('../../../data/random_faces/', image_size=64)

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
            nca_steps=[48, 96],
            lr = 1e-3,
            pool_size = 1024,
            num_damaged = 0,
            log_base_path = "tensorboard_logs",
            damage_radius = 3,
            device = device,
        )

    print(nca)

    try:
        trainer.train(batch_size=8, epochs=100000)
    except KeyboardInterrupt:
        nca.save("models/ConditionedNCA_{}.pt".format(datetime.now()))
