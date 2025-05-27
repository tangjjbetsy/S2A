import torch
import librosa
import numpy as np

class ChromaEvaluation():
    def __init__(self):
        pass


    def inference(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        data_in_np = audio.squeeze().cpu().numpy()
        feat_spec = librosa.feature.chroma_cqt(y=data_in_np, sr=24000).T
        return feat_spec


    def compute_dis(
        self,
        tar_input: np.array,
        gen_input: np.array,
        threshold=0.3,
    ) -> float:

        mask = np.asarray(tar_input > threshold, dtype=np.int32)
        tar = np.clip(tar_input, 0, 1)
        gen = np.clip(gen_input, 0.00001, 1-0.00001)
        
        len_s = min(tar.shape[0], gen.shape[0])
        
        dis = np.abs(tar[0:len_s] - gen[0:len_s]).sum() / mask.sum()
        return dis 