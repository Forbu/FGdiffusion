"""
Train G2PT + Fisher-Rao Flow Matching on NASA Knowledge Graph.

Uses continuous flow matching on the Fisher-Rao manifold for typed KG generation.
Each modality (edge indices, edge labels, node labels) flows on its own
Fisher-Rao sphere with Euler integration at inference.

Usage:
    python scripts/train_fisher_fm_kg_nasa.py
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from deepgraphgen.trainers.trainer_fisher_fm import TrainerKGFisherFM
from deepgraphgen.datasets import NasaKGDataset

torch.set_float32_matmul_precision("medium")

EDGE_TO_NODE_RATIO = 5
NB_NODES_TO_SAMPLE = 64
NB_CLASS = 15  # entity/relation types from trainer_llada_kg

# Vocabulary sizes
VOCAB_SIZE_EDGES = NB_NODES_TO_SAMPLE + 3       # node indices + pad + mask
VOCAB_SIZE_EDGE_LABELS = NB_CLASS + 1            # relation types + mask
VOCAB_SIZE_NODE_LABELS = NB_CLASS + 1            # entity types + mask

if __name__ == "__main__":
    model = TrainerKGFisherFM(
        vocab_size_edges=VOCAB_SIZE_EDGES,
        vocab_size_edge_labels=VOCAB_SIZE_EDGE_LABELS,
        vocab_size_node_labels=VOCAB_SIZE_NODE_LABELS,
        hidden_dim=384,
        nb_layer=6,
        heads=8,
        nb_max_node=NB_NODES_TO_SAMPLE,
        edges_to_node_ratio=EDGE_TO_NODE_RATIO,
        loss_type="ce",
        inference_steps=128,
    )

    training_dataset = NasaKGDataset(
        path_nodes="data_kg/nasa/nodes_nasa.parquet",
        path_edges="data_kg/nasa/edges_nasa.parquet",
        nb_nodes_to_sample=NB_NODES_TO_SAMPLE,
        edges_to_nodes_ratio=EDGE_TO_NODE_RATIO,
    )

    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    logger = pl.loggers.TensorBoardLogger("tb_logs/", name="fisher_fm_kg_nasa")

    callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="fisher_fm_kg_{epoch:02d}-{train_loss:.2f}",
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
