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
        # utt2spk[info['split']].append('{} {}\n'.format(uttid, "pianist_"+str(info['artist_id'])))

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

def main_score(db_root, tgt_root, use_sid=False, use_lid=False, use_perf=False):
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
    if use_perf:
        utt2perf = defaultdict(list)
    if use_sid:
        utt2spk = defaultdict(list)
    if use_lid:
        utt2lang= defaultdict(list)
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
        sid = "P"+str(info['artist_id'])
        lid = "A"+str(info['album_id'])
        
        utt2wav[info['split']].append('{} {}\n'.format("_".join([sid, lid, uttid]), os.path.join(db_root, 'ATEPP-s2a', info['midi_path'].replace(".mid", ".mp3"))))
        utt2text[info['split']].append('{} {}\n'.format("_".join([sid, lid, uttid]), os.path.join(db_root, 'ATEPP-s2a', info['score_path'] + ".midi")))
        if use_perf:
            utt2perf[info['split']].append('{} {}\n'.format("_".join([sid, lid, uttid]), os.path.join(db_root, 'ATEPP-s2a', info['midi_path'])))
        if use_sid:
            utt2spk[info['split']].append('{} {}\n'.format("_".join([sid, lid, uttid]), "P"+str(info['artist_id'])))
        if use_lid:
            utt2lang[info['split']].append('{} {}\n'.format("_".join([sid, lid, uttid]), "A"+str(info['album_id'])))

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
        
        if use_perf: 
            f_utt2perf = open(os.path.join(tgt_root, split, 'perf'), 'w')
            for item in utt2perf[split]:
                f_utt2perf.write(item)
            f_utt2perf.close()

        if use_sid:
            f_utt2spk = open(os.path.join(tgt_root, split, 'utt2spk'), 'w')
            for item in utt2spk[split]:
                f_utt2spk.write(item)
            f_utt2spk.close()
        
        if use_lid:
            f_utt2lang = open(os.path.join(tgt_root, split, 'utt2lang'), 'w')
            for item in utt2lang[split]:
                f_utt2lang.write(item)
            f_utt2lang.close()
        

if __name__ == '__main__':
    db_root = sys.argv[1]
    data = sys.argv[2]
    use_sid = sys.argv[3]
    use_lid = sys.argv[4]
    use_perf = sys.argv[5]
    main_score(db_root, data, use_sid, use_lid, use_perf)
