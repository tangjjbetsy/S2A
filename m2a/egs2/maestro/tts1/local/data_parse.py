import os
import sys
import glob
import json
import pandas as pd
from collections import defaultdict


"""
Parsing Maestro Database according to meta information, 
spliteing database into 'Train', 'Validation', and 'Test' subset,
who contains subsubset 'audio' and 'midi', implemented with soft link.
"""


def main(db_root, maestro_root):
    f_csv_dir = os.path.join(db_root, 'maestro-v3.0.0/maestro-v3.0.0.csv')
    df_dataset = pd.read_csv(f_csv_dir)
    dataset = defaultdict(dict)
    uttid_list = []
    for row in df_dataset.iterrows():
        prefix = os.path.basename(row[1]['midi_filename'])[:-5]
        dataset[prefix] = row[1]
        uttid_list.append(prefix)
    datalist_audio = glob.glob(os.path.join(db_root, 'maestro-v3.0.0/*/*.wav'))
    datalist_midi = glob.glob(os.path.join(db_root, 'maestro-v3.0.0/*/*.midi'))
    assert len(list(dataset.keys())) == len(datalist_audio)
    assert len(list(dataset.keys())) == len(datalist_midi)

    uttid_list.sort() # sort the uttid
    utt2wav = defaultdict(list)
    utt2text = defaultdict(list)
    utt2spk = defaultdict(list)
    for uttid in uttid_list:
        info = dataset[uttid]

        """
        tar_audio_dir = os.path.join(maestro_root, info['split'], 'audio')
        tar_midi_dir = os.path.join(maestro_root, info['split'], 'midi')
        if not os.path.exists(tar_audio_dir):
            os.makedirs(tar_audio_dir)
        if not os.path.exists(tar_midi_dir):
            os.makedirs(tar_midi_dir)

        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['audio_filename']),
            os.path.join(tar_audio_dir, os.path.basename(info['audio_filename']))
        ))
        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['midi_filename']),
            os.path.join(tar_midi_dir, os.path.basename(info['midi_filename']))
        ))
        """
        utt2wav[info['split']].append('{} {}\n'.format(uttid, os.path.join(db_root, 'maestro-v3.0.0', info['audio_filename'])))
        utt2text[info['split']].append('{} {}\n'.format(uttid, os.path.join(db_root, 'maestro-v3.0.0', info['midi_filename'])))
        # utt2spk[info['split']].append('{} {}\n'.format(uttid, 'piano'))

    for split in ['train', 'validation', 'test']:
        if not os.path.exists(os.path.join(maestro_root, split)):
            os.makedirs(os.path.join(maestro_root, split))

        f_utt2wav = open(os.path.join(maestro_root, split, 'wav.scp'), 'w')
        for item in utt2wav[split]:
            f_utt2wav.write(item)
        f_utt2wav.close()

        f_utt2text = open(os.path.join(maestro_root, split, 'text'), 'w')
        for item in utt2text[split]:
            f_utt2text.write(item)
        f_utt2text.close()

        """
        f_utt2spk = open(os.path.join(maestro_root, split, 'utt2spk'), 'w')
        for item in utt2spk[split]:
            f_utt2spk.write(item)
        f_utt2spk.close()
        """

def main_atepp(db_root, tgt_root):
    csv_dir = 'ATEPP-process/ATEPP-final-selection.csv' # Replace with the correct path to your CSV file
    f_csv_dir = os.path.join(db_root, csv_dir)
    df_dataset = pd.read_csv(f_csv_dir)
    dataset = defaultdict(dict)
    uttid_list = []
    for row in df_dataset.iterrows():
        prefix = os.path.basename(row[1]['midi_path'])[:-4]
        dataset[prefix] = row[1]
        uttid_list.append(prefix)
    # datalist_audio = glob.glob(os.path.join(db_root, 'ATEPP-selection/*/*.mp3'), recursive=True)
    # datalist_midi = glob.glob(os.path.join(db_root, 'ATEPP-selection/*/*.mid'), recursive=True)
    # assert len(list(dataset.keys())) == len(datalist_audio)
    # assert len(list(dataset.keys())) == len(datalist_midi)

    uttid_list.sort() # sort the uttid
    utt2wav = defaultdict(list)
    utt2text = defaultdict(list)
    # utt2spk = defaultdict(list)
    for uttid in uttid_list:
        info = dataset[uttid]

        """
        tar_audio_dir = os.path.join(maestro_root, info['split'], 'audio')
        tar_midi_dir = os.path.join(maestro_root, info['split'], 'midi')
        if not os.path.exists(tar_audio_dir):
            os.makedirs(tar_audio_dir)
        if not os.path.exists(tar_midi_dir):
            os.makedirs(tar_midi_dir)

        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['audio_filename']),
            os.path.join(tar_audio_dir, os.path.basename(info['audio_filename']))
        ))
        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['midi_filename']),
            os.path.join(tar_midi_dir, os.path.basename(info['midi_filename']))
        ))
        """            
        utt2wav[info['split']].append('{} {}\n'.format(uttid, os.path.join(db_root, 'ATEPP-finetune', info['audio_new_path'])))
        utt2text[info['split']].append('{} {}\n'.format(uttid, os.path.join(db_root, 'ATEPP-finetune', info['midi_path'])))
        # utt2spk[info['split']].append('{} {}\n'.format(uttid, 'piano'))

    for split in ['train', 'validation', 'test']:
        if not os.path.exists(os.path.join(tgt_root, split)):
            os.makedirs(os.path.join(tgt_root, split))

        f_utt2wav = open(os.path.join(tgt_root, split, 'wav.scp'), 'w')
        for item in utt2wav[split]:
            f_utt2wav.write(item)
        f_utt2wav.close()

        f_utt2text = open(os.path.join(tgt_root, split, 'text'), 'w')
        for item in utt2text[split]:
            f_utt2text.write(item)
        f_utt2text.close()

        """
        f_utt2spk = open(os.path.join(maestro_root, split, 'utt2spk'), 'w')
        for item in utt2spk[split]:
            f_utt2spk.write(item)
        f_utt2spk.close()
        """
        
def main_m2m(db_root, tgt_root, inference_path):
    f_csv_dir = os.path.join(db_root, 'ATEPP-s2a/ATEPP-s2a.csv')
    df_dataset = pd.read_csv(f_csv_dir)
    dataset = defaultdict(dict)
    uttid_list = []
    for row in df_dataset.iterrows():
        prefix = os.path.basename(row[1]['midi_path'])[:-4]
        dataset[prefix] = row[1]
        uttid_list.append(prefix)
    # datalist_audio = glob.glob(os.path.join(db_root, 'ATEPP-selection/*/*.mp3'), recursive=True)
    # datalist_midi = glob.glob(os.path.join(db_root, 'ATEPP-selection/*/*.mid'), recursive=True)
    # assert len(list(dataset.keys())) == len(datalist_audio)
    # assert len(list(dataset.keys())) == len(datalist_midi)

    uttid_list.sort() # sort the uttid
    utt2wav = defaultdict(list)
    utt2text = defaultdict(list)
    # utt2spk = defaultdict(list)
    for uttid in uttid_list:
        info = dataset[uttid]

        """
        tar_audio_dir = os.path.join(maestro_root, info['split'], 'audio')
        tar_midi_dir = os.path.join(maestro_root, info['split'], 'midi')
        if not os.path.exists(tar_audio_dir):
            os.makedirs(tar_audio_dir)
        if not os.path.exists(tar_midi_dir):
            os.makedirs(tar_midi_dir)

        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['audio_filename']),
            os.path.join(tar_audio_dir, os.path.basename(info['audio_filename']))
        ))
        os.system('ln -s {} {}'.format(
            os.path.join(db_root, info['midi_filename']),
            os.path.join(tar_midi_dir, os.path.basename(info['midi_filename']))
        ))
        """ 
        # inference_path = "/home/smg/v-jtbetsy/projects/lightning-hydra-template/logs/s2p_bert_no_activation_evaluate/runs/2024-05-16_11-45-03/predictions/"
        
        utt2wav[info['split']].append('{} {}\n'.format("_".join([uttid]), os.path.join(db_root, 'ATEPP-s2a', info['midi_path'].replace(".mid", ".mp3"))))
        if info['split'] == "test":
            utt2text[info['split']].append('{} {}\n'.format("_".join([uttid]), os.path.join(inference_path, os.path.basename(info['midi_path'])))) 
        else:
            utt2text[info['split']].append('{} {}\n'.format("_".join([uttid]), os.path.join(db_root, 'ATEPP-s2a', info['midi_path']))) 


    for split in ['train', 'validation', 'test']:
        if not os.path.exists(os.path.join(tgt_root, split)):
            os.makedirs(os.path.join(tgt_root, split))

        f_utt2wav = open(os.path.join(tgt_root, split, 'wav.scp'), 'w')
        for item in utt2wav[split]:
            f_utt2wav.write(item)
        f_utt2wav.close()

        f_utt2text = open(os.path.join(tgt_root, split, 'text'), 'w')
        for item in utt2text[split]:
            f_utt2text.write(item)
        f_utt2text.close()

        """
        f_utt2spk = open(os.path.join(maestro_root, split, 'utt2spk'), 'w')
        for item in utt2spk[split]:
            f_utt2spk.write(item)
        f_utt2spk.close()
        """

def main_inference(inference_path):
    files = glob.glob(f"{inference_path}/*.midi")
    uttid_list = []
    dataset = defaultdict(list)
    for file in files:
        prefix = file.split("/")[-1].split(".")[0]
        dataset[prefix] = file
        uttid_list.append(prefix)
    uttid_list.sort() # sort the uttid
    utt2text = defaultdict(list)
    for uttid in uttid_list:
       utt2text['test'].append('{} {}\n'.format("_".join([uttid]), dataset[uttid]))
       
    for split in ['test']:
        if not os.path.exists(os.path.join(tgt_root, split)):
            os.makedirs(os.path.join(tgt_root, split))
        f_utt2text = open(os.path.join(tgt_root, split, 'text'), 'w')
        for item in utt2text[split]:
            f_utt2text.write(item)
        f_utt2text.close()

# end def
if __name__ == '__main__':
    db_root = sys.argv[1]
    tgt_root = sys.argv[2]
    mode = sys.argv[3]
    
    if mode == "train":
        main_atepp(db_root, tgt_root)
    elif mode == "m2m":
        inference_path = sys.argv[4]
        main_m2m(db_root, tgt_root, inference_path)
    elif mode == "inference":
        inference_path = sys.argv[4]
        main_inference(inference_path)