'''
Modified from base implementation in:
    Sudhakaran, S., Najarro, E., & Risi, S. (2022).
    "Goal-guided neural cellular automata: Learning to control self-organising systems."

https://github.com/shyamsn97/controllable-ncas/blob/master/controllable_nca/experiments/morphing_image/trainer.py
'''


import math
import random
from typing import Any, Optional, Tuple  # noqa

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils.dataset import ConditioningDataset
from nca import ConditionedNCA
from sample_pool import SamplePool
from trainer import NCATrainer
from utils.utils import create_2d_circular_mask
from loss.loss import Loss


class ConditionedNCATrainer(NCATrainer):
    def __init__(
        self,
        nca: ConditionedNCA,
        target_dataset: ConditioningDataset,
        target_style_image: np.ndarray,
        nca_steps=[48, 96],
        lr: float = 2e-3,
        pool_size: int = 512,
        num_damaged: int = 0,
        log_base_path: str = "tensorboard_logs",
        damage_radius: int = 3,
        appearance_loss_type: str = 'OT',
        appearance_loss_weight: float = 1.0,
        content_loss_weight: float = 1.0,
        overflow_loss_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super(ConditionedNCATrainer, self).__init__(
            pool_size, num_damaged, log_base_path, device
        )
        self.target_dataset = target_dataset
        self.target_size = self.target_dataset.target_size

        self.nca = nca
        self.min_steps = nca_steps[0]
        self.max_steps = nca_steps[1]

        self.num_target_channels = self.target_size[0]
        self.image_size = self.target_size[-1]
        self.rgb = self.target_size[0] == 3
        self.damage_radius = damage_radius

        self.optimizer = torch.optim.Adam(self.nca.parameters(), lr=lr)
        self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [5000], gamma=0.3
        )

        print(device)
        self.loss = Loss(device=self.device,
                         content_loss_weight=content_loss_weight,
                         overflow_loss_weight=overflow_loss_weight,
                         appearance_loss_weight=appearance_loss_weight,
                         appearance_loss_type=appearance_loss_type,
                         target_style_image=target_style_image)

    def emit_metrics(self, i: int, batch, outputs, targets, loss, metrics={}):
        with torch.no_grad():
            self.train_writer.add_scalar("loss", loss, i)
            self.train_writer.add_scalar("log10(loss)", math.log10(loss), i)
            self.train_writer.add_images(
                "batch", self.to_rgb(batch), i, dataformats="NCHW"
            )
            self.train_writer.add_images(
                "outputs", self.to_rgb(outputs), i, dataformats="NCHW"
            )
            self.train_writer.add_images(
                "targets", self.to_rgb(targets), i, dataformats="NCHW"
            )
            for k in metrics:
                self.train_writer.add_scalar(k, metrics[k], i)

    def damage(self, batch):
        size = batch.size(0)
        for i in range(self.num_damaged):
            mask = create_2d_circular_mask(
                self.image_size, self.image_size, radius=self.damage_radius
            )
            batch[max(size - i - 1, 0)][:, mask] *= 0.0
        return batch

    def sample_batch(self, sampled_indices, sample_pool) -> Tuple[Any, Any]:
        """
        Returns batch + targets

        Returns:
            Tuple[Any, Any]: [description]
        """
        batch = sample_pool[sampled_indices]
        for i in range(len(sampled_indices)):
            if batch[i] is None:
                batch[i] = self.nca.generate_seed(1)[0].to(self.device)
            elif torch.sum(self.nca.alive(batch[i].unsqueeze(0))) == 0.0:
                batch[i] = self.nca.generate_seed(1)[0].to(self.device)
        batch = torch.stack(batch)
        return batch

    def sample_targets(self, sampled_indices):
        batch_len = len(sampled_indices)
        num_targets = len(self.target_dataset)
        random_indices = np.random.choice(num_targets, batch_len, replace=True)
        return self.target_dataset[random_indices]

    def train_batch(self, batch, targets):
        num_steps = random.randint(self.min_steps, self.max_steps)
        batch = self.nca.grow(batch, num_steps=num_steps, goal=targets)

        loss_input_dict = {'target_images': targets, 'nca_state': batch,
                           'generated_images': batch[:, : self.num_target_channels, :, :]}
        loss, loss_summary = self.loss(loss_input_dict)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.nca.parameters(), 100.0)
        for p in self.nca.parameters():
            if p.grad is not None:
                p.grad /= torch.norm(p.grad) + 1e-10
        self.optimizer.step()
        self.lr_sched.step()
        grad_dict = {}
        for n, W in self.nca.named_parameters():
            if W.grad is not None:
                grad_dict["{}_grad".format(n)] = float(torch.sum(W.grad).item())

        return (
            batch.detach(),
            loss.item(),
            {
                "loss": loss.item(),
                **loss_summary,
                "log10loss": math.log10(loss.item() + 1e-5),
                **grad_dict,
            },
        )

    def update_pool(self, idxs, outputs, targets):
        self.pool[idxs] = outputs.detach()

    def train(self, batch_size, epochs, *args, **kwargs):
        self.pool = SamplePool(self.pool_size)
        bar = tqdm.tqdm(range(epochs))
        for i in bar:
            idxs = random.sample(range(len(self.pool)), batch_size)

            with torch.no_grad():
                targets = self.sample_targets(idxs).to(self.device)
                batch = self.sample_batch(idxs, self.pool).to(self.device)
                batch[:2] = self.nca.generate_seed(2)

            outputs, loss, metrics = self.train_batch(batch, targets)
            # train more
            outputs, loss, metrics = self.train_batch(outputs, targets)
            # Place outputs back in the pool.
            self.update_pool(idxs, outputs, targets)
            description = "--".join(["{}:{}".format(k, metrics[k]) for k in metrics])
            description = f"Epoch {i}/{epochs}: {description}"
            bar.set_description(description)
            self.emit_metrics(i, batch, outputs, targets, loss, metrics=metrics)
