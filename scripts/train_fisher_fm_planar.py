"""
Train G2PT + Fisher-Rao Flow Matching on planar graphs.

Uses continuous flow matching on the Fisher-Rao manifold (positive sphere S⁺_d)
instead of discrete diffusion (LLaDA/score-based). The endpoint parametrization
predicts x_1 directly, with Euler integration on the sphere at inference.

Usage:
    python scripts/train_fisher_fm_planar.py
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from deepgraphgen.trainers.trainer_fisher_fm import TrainerG2PTFisherFM
from deepgraphgen.datasets import SpectreGraphDataset, SpectreIterableDataset

torch.set_float32_matmul_precision("medium")

EDGES_TO_NODE_RATIO = 5
NB_MAX_NODE = 64
VOCAB_SIZE = NB_MAX_NODE + 3  # node indices + pad + mask

if __name__ == "__main__":
    model = TrainerG2PTFisherFM(
        vocab_size=VOCAB_SIZE,
        hidden_dim=384,
        nb_layer=6,
        heads=8,
        nb_max_node=NB_MAX_NODE,
        edges_to_node_ratio=EDGES_TO_NODE_RATIO,
        loss_type="ce",        # "ce" (recommended), "velocity", or "geodesic"
        inference_steps=128,   # Euler steps for inference
        use_time_emb=True,
    )

    training_dataset = SpectreGraphDataset(
        dataset_name="planar",
        download_dir="scripts/datasets",
        edges_to_node_ratio=EDGES_TO_NODE_RATIO,
    )

    iterdataset = SpectreIterableDataset(dataset=training_dataset)
    training_dataloader = DataLoader(iterdataset, batch_size=32)

    logger = pl.loggers.TensorBoardLogger("tb_logs/", name="fisher_fm_planar")

    callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="fisher_fm_planar_{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
        dirpath="models/",
        every_n_train_steps=3000,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_time={"hours": 14},
        logger=logger,
        accumulate_grad_batches=4,
        callbacks=[callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, training_dataloader)
