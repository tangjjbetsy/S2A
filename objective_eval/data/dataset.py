import glob
import os

import librosa
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        data_tag: str,
        data_dir: str,
        data_suffix: str = '.wav',
        sample_rate: int = 24000,
    ):
        self.data_tag = data_tag
        self.data_dir = data_dir
        self.data_suffix = data_suffix
        self.sample_rate = sample_rate

        self.data_list = glob.glob(
            os.path.join(self.data_dir, '*' + self.data_suffix)
        )
        self.data_list.sort()

    def __getitem__(
        self,
        index,
    ):
        wav_dir = self.data_list[index]
        wav_name = os.path.basename(wav_dir)[:-4]
        wav = librosa.core.load(wav_dir, sr=self.sample_rate)[0]
        return {'wav': wav, 'wav_name': wav_name}

    def __len__(
        self,
    ):
        return len(self.data_list)
