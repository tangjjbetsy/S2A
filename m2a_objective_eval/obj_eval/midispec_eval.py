import torch
import numpy as np

from supportive.spectrogram.audio import Audio
from supportive.spectrogram.hparams import Hparams_class


class MIDISpecEvaluation():
    def __init__(self):
        self.hparams_ins = Hparams_class()
        self.audio = Audio(self.hparams_ins)

    def inference(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        data_in_np = audio.squeeze().cpu().numpy()
        mel_spectrogram = self.audio.melspectrogram(data_in_np).astype(np.float32).T  # (1223, 128)
        # mel_spectrogram.tofile(str(os.path.join(args.output_dir, data_info[0].split(',')[1] + '.npy')), format="<f4")
        return mel_spectrogram


    def compute_dis(
        self,
        tar_input: np.array,
        gen_input: np.array,
        threshold=0.3,
    ) -> float:
        #
        # __import__('ipdb').set_trace()
        # t_min = min(tar_input.shape[0], gen_input.shape[0])
        # tar_input = tar_input[:t_min, :]
        # gen_input = gen_input[:t_min, :]
        # mask = np.asarray(tar_input > threshold, dtype=np.int32)
        # tar = np.clip(tar_input, 0, 1)
        # gen = np.clip(gen_input, 0.00001, 1-0.00001)
        len_s = min(tar_input.shape[0], gen_input.shape[0])

        dis = np.sqrt(((tar_input[0:len_s] - gen_input[0:len_s]) ** 2).sum() / (tar_input[0:len_s] ** 2).sum())
        return dis
