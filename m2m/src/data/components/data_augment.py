# Copyright (c) 2024 Jingjing Tang
#
# -*- coding:utf-8 -*-
# @Script: data_augment.py
# @Author: Jingjing Tang
# @Email: tangjingjingbetsy@gmail.com
# @Create At: 2024-05-22 20:55:47
# @Last Modified By: Jingjing Tang
# @Last Modified At: 2024-05-23 13:40:13
# @Description: This is description.

import numpy as np
import os
import glob
from tqdm import tqdm

class AugmentTokenSeq():
    def __init__(self):
        pass
    
    def _shift_key(self, seq, semitone=1):
        for i in range(len(seq)):
            if seq[i][0] not in [0,1,2,3]:
                seq[i][0] += semitone
                seq[i][0] = self._clip_feature(seq[i][0], [4, 91])
                
                assert seq[i][0] <= 91
                assert seq[i][0] >= 4
        return seq

    def _shift_velocity(self, seq, shift=1):
        for i in range(len(seq)):
            if seq[i][1] not in [0,1,2,3]:
                seq[i][1] += shift
                seq[i][1] = self._clip_feature(seq[i][1], [4, 67])
                
                assert seq[i][1] <= 67
                assert seq[i][1] >= 4
        return seq

    def _clip_feature(self, token, bounds):
        token = bounds[1] if token > bounds [1] else token
        token = bounds[0] if token < bounds [0] else token
        return token


    def __call__(self, path, key=True, vel=True):
        data = np.load(path, allow_pickle=True).item()
        filename = os.path.basename(path)
        if key:
            i = 0
            while i == 0:
                i = np.random.randint(-3, 3)
            data['score_seq'] = self._shift_key(data['score_seq'], semitone=i)
            np.save(path.replace(filename, f"{data['perf_id']}_key_{i}.npy"), data)
        
        if vel:
            i = 0
            while i == 0:
                i = np.random.randint(-6, 6)
            data['perf_seq'] = self._shift_velocity(data['perf_seq'], shift=i)
            np.save(path.replace(filename, f"{data['perf_id']}_vel_{i}.npy"), data)
        return

    def clear_all():
        for split in ['train', 'validation', 'test']:
            files = glob.glob(f"data/{split}" + "/*_*_*.npy")
            for file in files:
                os.remove(file)
    


#Not apply to the compact file "data.npz", seperate files only
def calculate_note_density(seq):
    seq_len = len(seq)
    
    if seq[0][0] == 1:
        seq_len -= 1 
        bar_s = seq[1][-1] 
    else:
        bar_s = seq[0][-1]
    
    i = 1
    while seq[-i][-1] in [0, 2]: 
        seq_len -= 1 
        i += 1
        
    bar_e = seq[-i][-1] 
    
    return np.round(seq_len / (bar_e - bar_s))


if  __name__ == "__main__":
    # for split in ['train', 'validation', 'test']:
    files = glob.glob(f"data/train" + "/*.npy")
    for path in tqdm(files):
        AugmentTokenSeq(path)
    
    # clear_all()