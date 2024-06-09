from pathlib import Path

import pytest
import torch

from src.data.expression_datamodule import ExpressionDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_expression_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = ExpressionDataModule(data_dir=data_dir, batch_size=batch_size, load_data="data/data.npz")
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "data.npz").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert (
        dm.train_dataloader()
        and dm.val_dataloader()
        and dm.test_dataloader()
        and dm.predict_dataloader()
    )

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 9468

    batch = next(iter(dm.train_dataloader()))
    x, mask, y, style, idx = batch
    assert len(x) == batch_size
    assert len(x[0]) == 256
    assert len(y) == batch_size
    assert len(y[0]) == 256
    assert len(mask) == batch_size
    assert len(style) == batch_size
    assert len(idx) == batch_size
