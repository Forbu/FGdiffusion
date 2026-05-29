"""
Train G2PT + LLaDA discrete diffusion on random trees.

Usage:
    python scripts/train_llada_tree.py
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from deepgraphgen.trainers.trainer_llada import TrainerG2PTLLaDA
from deepgraphgen.datasets import TreeDataset, SpectreIterableDataset

torch.set_float32_matmul_precision("medium")

EDGES_TO_NODE_RATIO = 3

if __name__ == "__main__":
    model = TrainerG2PTLLaDA(
        nb_max_node=100,
        hidden_dim=768,
        nb_layer=12,
        heads=12,
        edges_to_node_ratio=EDGES_TO_NODE_RATIO,
    )

    training_dataset = TreeDataset(10000, 10, edges_to_nodes_ratio=EDGES_TO_NODE_RATIO)
    iterdataset = SpectreIterableDataset(dataset=training_dataset)
    training_dataloader = DataLoader(iterdataset, batch_size=32)

    logger = pl.loggers.TensorBoardLogger("tb_logs/", name="llada_tree")

    callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="tree_llada_{epoch:02d}-{train_loss:.2f}",
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
