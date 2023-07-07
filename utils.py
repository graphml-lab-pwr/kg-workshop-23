import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


def prepare_for_visualization(
    embeddings: np.ndarray, labels: list[str], log_dir: Path = Path("logs")
) -> None:
    assert len(embeddings) == len(
        labels
    ), "Embeddings and labels must have the same length"
    assert len(embeddings.shape) == 2, "Embeddings must be a 2D array"
    # Assume a logdir exists
    log_dir.mkdir(parents=True, exist_ok=True)
    # 1) Save a .tsv with labels, line by line
    METADATA_FNAME = "meta.tsv"
    with open(os.path.join(log_dir, METADATA_FNAME), "w+") as f:
        for label in labels:
            f.write("{}\n".format(label))

    # 2) Save the embedding in a checkpoint.
    weights = tf.Variable(embeddings, name="my weights var")
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # 3) Configure the projector
    config = projector.ProjectorConfig()
    config.embeddings.add(
        tensor_name="embedding/.ATTRIBUTES/VARIABLE_VALUE",
        metadata_path=METADATA_FNAME,
    )
    projector.visualize_embeddings(log_dir, config)
