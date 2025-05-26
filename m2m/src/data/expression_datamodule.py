from .components import ExpressionDataset
from typing import Any, Dict, Optional, Tuple

import glob
import os
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

class DealDataset(Dataset):
    def __init__(self, x_data, mask_data, y_data, style, idx, output_features):
        self.x_data = torch.LongTensor(x_data)
        # self.x_data = torch.FloatTensor(x_data)
        self.mask_data = torch.FloatTensor(mask_data)
        self.y_data = torch.LongTensor(y_data)
        # self.y_data = torch.FloatTensor(y_data)
        self.style = torch.LongTensor(style)
        self.idx = idx
        self.len = self.x_data.shape[0]
        
        # what features to be used as prediction target
        y_features = output_features
        # y_features = ["Pitch", "Velocity", "Duration", "IOI", "Position", "Bar"]
        
        self.y_data = self.y_data[:, :, self.get_feature_index(y_features)] 
    
    def __getitem__(self, index):
        return self.x_data[index], self.mask_data[index], self.y_data[index],  self.style[index], self.idx[index]

    def __len__(self):
        return self.len

    def get_feature_index(self, feature_names, mtype="perf"):
        feature_list = ["Pitch", "Velocity", "Duration", "IOI", "Position", "Bar"]        
        feature_idx = [feature_list.index(name) for name in feature_names]
        
        if len(feature_names) == 0:
            return feature_idx[0]
        else:
            return feature_idx 

class ExpressionDataModule(LightningDataModule):
    """`LightningDataModule` for the piano expression-related dataset.
    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...
    ```
    """

    def __init__(
        self,
        data_dir: str = "data/",
        output_data: str = "data/data.npz",
        load_data: str = "from_files",
        output_tokenizer: str = None,
        load_tokenizer: str = None,
        tokenizer_conf: dict = None,
        mode: str = "token",
        csv_file: str = None,
        data_folders: list[str] = [],
        feature_list: list[str] = [],
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        num_workers: int = 0,
        max_len: int = 256,
        styles: int = 0,
        alignment: bool = False,
        score: bool = False,
        transcribe: bool = False,
        split: bool = False,
        padding: bool = True,
        compact: bool = False,
        pin_memory: bool = False,
        output_features: list = [],
        stage: str = "train"
    ) -> None:
        """Initialize a `ExpressionDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of styles.

        :return: The number of Expresstion styles.
        """
        return self.hparams.styles

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if (os.path.isfile(self.hparams.output_data) == False) and \
            (os.path.isdir(os.path.join(self.hparams.data_dir, "train")) == False):
            self.data = ExpressionDataset(self.hparams)
        else:
            if self.hparams.load_data == "from_npz":
                self.data = np.load(self.hparams.output_data, allow_pickle=True)
            else:
                self.data = dict()
                train_dir = os.path.join(self.hparams.data_dir, "train")
                validation_dir = os.path.join(self.hparams.data_dir, "validation")
                test_dir = os.path.join(self.hparams.data_dir, "test")
                
                for split in ['train', 'validation', 'test']:
                    files = glob.glob(eval(split + "_dir") + "/*.npy")
                    self.data[split] = [np.load(file, allow_pickle=True).item() for file in files]
        
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # assume using score as input and performance as output
        if not self.data_train and not self.data_val and not self.data_test:
            tmp = dict()
            if self.hparams['stage'] == 'train':
                stage_list = ['train', 'validation', 'test']
            else:
                stage_list = ['test']
                
            for split in stage_list:
                tmp[split] = DealDataset([data['score_seq'] for data in self.data[split]],
                                   [data['perf_mask'] for data in self.data[split]],
                                   [data['perf_seq'] for data in self.data[split]],
                                   [data['performer_id'] for data in self.data[split]],
                                   [data['perf_id'] for data in self.data[split]],
                                   self.hparams.output_features)
            
            ##### Resplit the dataset #####
            # dataset = ConcatDataset(datasets=[tmp['train'], tmp['validation'], tmp['test']])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            
            if self.hparams['stage'] == 'train':
                self.data_train = tmp['train']
                self.data_val = tmp['validation']
                self.data_test = tmp['test']
            else:
                self.data_test = tmp['test']
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ExpressionDataModule()
