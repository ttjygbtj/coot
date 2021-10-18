"""
Dataset utilities.
"""

from typing import Any, Callable, List, Optional

import paddle

from nntrainer import trainer_configs, typext


class DataSplitConst(typext.ConstantHolder):
    """
    Store dataset splits.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def create_loader(dataset: paddle.io.Dataset, cfg: trainer_configs.BaseDatasetConfig, batch_size: int, *,
                  collate_fn: Optional[Callable[[List[Any]], Any]] = None) -> paddle.io.DataLoader:
    """
    Create torch dataloader from torch dataset.

    Args:
        dataset: Dataset.
        cfg: Dataset configuration.
        batch_size: Batch size.
        collate_fn: Collation function to be used to stack the data into batches.

    Returns:
    """
    return paddle.io.DataLoader(
        dataset, batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers,
        drop_last=cfg.drop_last, collate_fn=collate_fn)  # type: ignore
