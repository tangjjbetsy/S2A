import argparse
import json
import os
from random import shuffle

import torch
from torch.utils.data import DataLoader

from data import AudioDataset, URMPDataset
from obj_eval import ChromaEvaluation, MIDISpecEvaluation, TimbreEvaluation


FEAT_EXTR = {
    'timbre': TimbreEvaluation,
    'midispec': MIDISpecEvaluation,
    'chroma': ChromaEvaluation,
}


def main():

    # argparse definition
    parser = argparse.ArgumentParser(description='Objective Evaluation on MIDI-to-wav')
    parser.add_argument('--database', 
                        type=str,
                        choices=['maestro', 'nii-urmp'],
                        default='maestro', 
                        help='the type of database')
    parser.add_argument('--feat', 
                        type=str,
                        choices=['f0', 'chroma', 'midispec', 'timbre'],
                        default='chroma', 
                        help='the type of feature to extract')
    parser.add_argument('--data-json', 
                        type=str,
                        default=None, 
                        help='the directory of data information json file')
    args = parser.parse_args()
    
    # feature extraction
    feature_extractor_class = FEAT_EXTR[args.feat]
    model = feature_extractor_class()

    data_f = open(args.data_json, 'r')
    data_json = json.load(data_f)
    exp_sys_list = data_json['exp_sys']
    data_dir = data_json['data_dir']
    for sys_name in exp_sys_list:
        print('Processing {}-{}'.format(args.feat, sys_name))
        wav_dir = os.path.join(
            data_dir, sys_name
        )
        dataset = AudioDataset(sys_name, wav_dir)
        dataloader = DataLoader(dataset, shuffle=False)

        save_dir = os.path.join('output', args.feat, sys_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for _, batch in enumerate(dataloader):
            wav = batch['wav']
            wav_name = batch['wav_name'][0]
            feat = model.inference(wav)  # feat (T, F)
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            feat_dir = os.path.join(
                save_dir, wav_name.split("_")[-1] + '.npy'
            )
            feat.tofile(feat_dir, format="<f4")


if __name__ == '__main__':
    main()
