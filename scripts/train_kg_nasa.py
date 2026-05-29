"""
Train G2PT + LLaDA on NASA Knowledge Graph.

Usage:
    python scripts/train_kg_nasa.py
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from deepgraphgen.trainers.trainer_llada_kg import TrainerG2PTKG
from deepgraphgen.datasets import NasaKGDataset

torch.set_float32_matmul_precision("medium")

EDGE_TO_NODE_RATIO = 5
NB_NODES_TO_SAMPLE = 64

if __name__ == "__main__":
    model = TrainerG2PTKG(
        nb_max_node=NB_NODES_TO_SAMPLE,
        hidden_dim=386,
        nb_layer=6,
        heads=8,
        edges_to_node_ratio=EDGE_TO_NODE_RATIO,
    )

    training_dataset = NasaKGDataset(
        path_nodes="data_kg/nasa/nodes_nasa.parquet",
        path_edges="data_kg/nasa/edges_nasa.parquet",
        nb_nodes_to_sample=NB_NODES_TO_SAMPLE,
        edges_to_nodes_ratio=EDGE_TO_NODE_RATIO,
    )

    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    logger = pl.loggers.TensorBoardLogger("tb_logs/", name="kg_nasa")

    callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="nasakg_llada_{epoch:02d}-{train_loss:.2f}",
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
        accumulate_grad_batches=1,
        callbacks=[callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, training_dataloader)
