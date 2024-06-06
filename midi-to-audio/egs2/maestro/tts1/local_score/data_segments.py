import os
import argparse
import librosa
import pretty_midi
import scipy
import scipy.io.wavfile
import scipy.sparse
import numpy as np
import multiprocessing
"""

"""

FRAME_LENGTH_MS = 50
FRAME_SHIFT_MS = 12

def create_mono_note_sequences_from_multi(mono, instruments):
    """Add all notes to the mono notes list
    Args:
        mono (pretty_midi.Instrument): Target instrument
        instruments (List): List of instruments used in multiple tracks
    """
    for instrument in instruments[1:]:
        mono.notes += instrument.notes
        mono.notes = sorted(mono.notes, key=lambda x: x.start)

def create_mono_control_sequences_from_multi(mono, instruments):
    """Add all the control changes to the mono track

    Args:
        mono (pretty_midi.Instrument): the target mono track
        instruments (List): list of the multichannel tracks
    """
    for instrument in instruments[1:]:
        mono.control_changes += instrument.control_changes
        mono.control_changes = list(set(mono.control_changes)) #Remove repeated control changes
        mono.control_changes = sorted(mono.control_changes, key=lambda x: x.time)

def pad_or_cut_midi(seq, require_len):
        """Padding or cutting the sequence to the required length

        Args:
            seq (list): input sequence
            require_len (int): specify expected length after padding or cutting

        Returns:
            list: sequence with expected length
        """
        if seq.shape[1] >= require_len:
            return seq[:, 0:require_len]
        else:
            return np.concatenate([seq, np.zeros((int(seq.shape[0]), int(require_len-seq.shape[1])))], axis= 1)  

def pad_or_cut_wav(seq, require_len):
        """Padding or cutting the sequence to the required length

        Args:
            seq (list): input sequence
            require_len (int): specify expected length after padding or cutting

        Returns:
            list: sequence with expected length
        """
        if len(seq) >= require_len:
            return seq[0:require_len]
        else:
            return np.concatenate([seq, np.zeros(int(require_len - len(seq)))], axis = 0)  


def segment_func(midi_wav_args_zip):
    midi_item = midi_wav_args_zip[0]
    wav_item = midi_wav_args_zip[1]
    args = midi_wav_args_zip[2]

    wav_id, wav_dir = wav_item.strip().split()
    midi_id, midi_dir = midi_item.strip().split()
    assert wav_id == midi_id

    up_sample_rate = FRAME_SHIFT_MS / 1000 * args.sample_rate
    frame_length_point = int(FRAME_LENGTH_MS / 1000 * args.sample_rate) #1200
    frame_shift_point = int(FRAME_SHIFT_MS / 1000 * args.sample_rate) #288
    segment_length_point = (args.num_segment_frame - 1) * frame_shift_point + frame_length_point #242112

    try:
        mid_ob = pretty_midi.PrettyMIDI(midi_dir)
    except:
        if not os.path.isfile(midi_dir):
            raise ValueError('cannot find midifile from {}'.format(wav_dir))
        else:
            raise ValueError('cannot read midifile from {}'.format(wav_dir))
    if len(mid_ob.instruments) > 1:
        print("Track has >1 instrument %s" % (midi_dir))
    midi = mid_ob.get_piano_roll(fs=args.sample_rate/up_sample_rate)

    try:
        wav = librosa.core.load(wav_dir, sr=args.sample_rate)[0]
    except:
        raise ValueError('cannot read waveform from {}'.format(wav_dir))
    # time = librosa.get_duration(wav, sr=args.sample_rate)

    utt_id_list = []
    print('split uttid: {}'.format(wav_id))
    i = 0
    while (i+1)*segment_length_point < wav.shape[0]:
        utt_id = "{}_{}".format(wav_id, i)
        wav_segment_dir = os.path.join(args.wav_segments_dir, utt_id + '.wav')
        midi_segment_dir = os.path.join(args.text_segments_dir, utt_id + '.npz')
        # wav_segments_list.append('{} {}\n'.format(utt_id, wav_segment_dir))
        # midi_segments_list.append('{} {}\n'.format(utt_id, midi_segment_dir))
        utt_id_list.append(utt_id)

        wav_begin_point = int(args.num_segment_frame * frame_shift_point * i)
        wav_end_point = int(args.num_segment_frame * frame_shift_point * (i + 1) + frame_length_point)
        wav_segment = wav[wav_begin_point:wav_end_point]
        midi_segment = midi[:, int(args.num_segment_frame*i):int(args.num_segment_frame*(i+1))]
        sparse_midi_segment = scipy.sparse.csc_matrix(midi_segment)
        
        if not os.path.isdir(os.path.dirname(wav_segment_dir)):
            os.makedirs(os.path.dirname(wav_segment_dir))
        if not os.path.isdir(os.path.dirname(midi_segment_dir)):
            os.makedirs(os.path.dirname(midi_segment_dir))
        scipy.io.wavfile.write(wav_segment_dir, args.sample_rate, wav_segment)
        scipy.sparse.save_npz(midi_segment_dir, sparse_midi_segment)

        i += 1
    return utt_id_list

def segment_func_new(midi_wav_args_zip):
    midi_item = midi_wav_args_zip[0]
    wav_item = midi_wav_args_zip[1]
    args = midi_wav_args_zip[2]
    wav_seg_points = midi_wav_args_zip[3]
    midi_seg_points = midi_wav_args_zip[4]
    if args.use_perf:
        perf_item = midi_wav_args_zip[5]
        perf_seg_points = midi_wav_args_zip[6]

    wav_id, wav_dir = wav_item.strip().split()
    midi_id, midi_dir = midi_item.strip().split()
    if args.use_perf:
        perf_id, perf_dir = perf_item.strip().split()
    assert wav_id == midi_id


    up_sample_rate = FRAME_SHIFT_MS / 1000 * args.sample_rate #288
    frame_length_point = int(FRAME_LENGTH_MS / 1000 * args.sample_rate) #1200
    frame_shift_point = int(FRAME_SHIFT_MS / 1000 * args.sample_rate) #288
    segment_length_point = (args.num_segment_frame - 1) * frame_shift_point + frame_length_point #231312

    try:
        mid_ob = pretty_midi.PrettyMIDI(midi_dir)
        if args.use_perf:
            perf_ob = pretty_midi.PrettyMIDI(perf_dir)
    except:
        if not os.path.isfile(midi_dir):
            raise ValueError('cannot find midifile from {}'.format(wav_dir))
        else:
            raise ValueError('cannot read midifile from {}'.format(wav_dir))
    
    if len(mid_ob.instruments) > 1:
        mono = mid_ob.instruments[0]
        mono.program = 0

        create_mono_note_sequences_from_multi(mono, mid_ob.instruments)
        create_mono_control_sequences_from_multi(mono, mid_ob.instruments)

        new_midi = pretty_midi.PrettyMIDI()
        new_midi.instruments.append(mono)
        mid_ob = new_midi
        # print("Track has >1 instrument %s" % (midi_dir))
    midi = mid_ob.get_piano_roll(fs=args.sample_rate/up_sample_rate)
    if args.use_perf:
        perf = perf_ob.get_piano_roll(fs=args.sample_rate/up_sample_rate)

    try:
        wav = librosa.core.load(wav_dir, sr=args.sample_rate)[0]
    except:
        raise ValueError('cannot read waveform from {}'.format(wav_dir))
    # time = librosa.get_duration(wav, sr=args.sample_rate)

    utt_id_list = []
    print('split uttid: {}'.format(wav_id))
    
    i = 0
    n = 0
    wav_seg_points = [int(k * args.sample_rate) for k in wav_seg_points]
    midi_seg_points = [int(k * args.sample_rate / up_sample_rate) for k in midi_seg_points]
    if args.use_perf:
        perf_seg_points = [int(k * args.sample_rate / up_sample_rate) for k in perf_seg_points]
    
   
    while (i+1) < len(wav_seg_points) - 1:
        utt_id = "{}_{}".format(wav_id, n)
        wav_segment_dir = os.path.join(args.wav_segments_dir, utt_id + '.wav')
        midi_segment_dir = os.path.join(args.text_segments_dir, utt_id + '.npz')
        if args.use_perf:
            perf_segment_dir = os.path.join(args.perf_segments_dir, utt_id + '.npz')
        # wav_segments_list.append('{} {}\n'.format(utt_id, wav_segment_dir))
        # midi_segments_list.append('{} {}\n'.format(utt_id, midi_segment_dir))
        utt_id_list.append(utt_id)
    
        # wav_begin_point = int(args.num_segment_frame * frame_shift_point * i)
        # wav_end_point = int(args.num_segment_frame * frame_shift_point * (i + 1) + frame_length_point)
        
        if (wav_seg_points[i] != -1) & (wav_seg_points[i+1] != -1):
            wav_segment = wav[wav_seg_points[i]:wav_seg_points[i + 1]]
            midi_segment = midi[:, midi_seg_points[i]:midi_seg_points[i + 1]]
                
            ######## Padding or Cut to Required Length #########
            # wav_segment = pad_or_cut_wav(wav_segment, int(args.num_segment_frame * frame_shift_point + frame_length_point))
            # midi_segment = pad_or_cut_midi(midi_segment, int(args.num_segment_frame))            
            
            sparse_midi_segment = scipy.sparse.csc_matrix(midi_segment)
            if not os.path.isdir(os.path.dirname(wav_segment_dir)):
                os.makedirs(os.path.dirname(wav_segment_dir))
            if not os.path.isdir(os.path.dirname(midi_segment_dir)):
                os.makedirs(os.path.dirname(midi_segment_dir))
            if args.use_perf:
                if not os.path.isdir(os.path.dirname(perf_segment_dir)):
                    os.makedirs(os.path.dirname(perf_segment_dir))
                
            scipy.io.wavfile.write(wav_segment_dir, args.sample_rate, wav_segment)
            scipy.sparse.save_npz(midi_segment_dir, sparse_midi_segment)
            
            i += 1
            n += 1
            
        else:
            i += 1
            continue
    return utt_id_list


def main():
    # parser
    parser = argparse.ArgumentParser(description='generate segments')
    parser.add_argument('--wav_dir', type=str, default='',
                        help='directory of wav.scp')
    parser.add_argument('--wav_segments_dir', type=str, default='',
                        help='directory of segments for wav')
    parser.add_argument('--text_dir', type=str, default='',
                        help='directory of text (directory of score midi)')
    parser.add_argument('--text_segments_dir', type=str, default='',
                        help='directory of segments for text')
    parser.add_argument('--perf_dir', type=str, default='',
                        help='directory of text (directory of perf midi)')
    parser.add_argument('--perf_segments_dir', type=str, default='',
                        help='directory of segments for text')
    parser.add_argument('--segmentation_points_dir', type=str, default='',
                        help='directory of time points to segment audios and scores')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='number of sample rate')
    parser.add_argument('--num_segment_frame', type=float, default=800,
                        help='number of frames in each segment')
    parser.add_argument('--begin_cut', type=float, default=2,
                        help='the beginning of the frame')
    parser.add_argument('--end_cut', type=float, default=2,
                        help='the end of the frame')
    
    args = parser.parse_args()
    
    f_wav = open(args.wav_dir, 'r')
    lines_wav = f_wav.readlines()

    f_midi = open(args.text_dir, 'r')
    lines_midi = f_midi.readlines()
    
    if args.use_perf:
        f_perf = open(args.perf_dir, 'r')
    lines_perf = f_perf.readlines()

    wav_segments_list = []
    midi_segments_list = []
    if args.use_perf:
        perf_segments_list = []
    segmentation_points = np.load(args.segmentation_points_dir, allow_pickle=True).item()

    midi_wav_args_list = []
    if args.use_perf:
        for wav_item, midi_item, perf_item in zip(lines_wav, lines_midi, lines_perf):
            wav_id, wav_dir = wav_item.strip().split()
            midi_id, midi_dir = midi_item.strip().split()
            perf_id, perf_dir = perf_item.strip().split()
            wav_id = wav_id.split("_")[-1]
            midi_id = midi_id.split("_")[-1]
            perf_id = perf_id.split("_")[-1]
            wav_seg_points = segmentation_points['audio'][str(wav_id)]
            midi_seg_points = segmentation_points['score'][str(midi_id)]  
            perf_seg_points = segmentation_points['audio'][str(perf_id)]  
            assert wav_id == midi_id
            midi_wav_args_list.append([midi_item, wav_item, args, wav_seg_points, midi_seg_points, perf_item, perf_seg_points])
    else:
        for wav_item, midi_item in zip(lines_wav, lines_midi):
            wav_id, wav_dir = wav_item.strip().split()
            midi_id, midi_dir = midi_item.strip().split()
            wav_id = wav_id.split("_")[-1]
            midi_id = midi_id.split("_")[-1]
            wav_seg_points = segmentation_points['audio'][str(wav_id)]
            midi_seg_points = segmentation_points['score'][str(midi_id)]  
            assert wav_id == midi_id
            midi_wav_args_list.append([midi_item, wav_item, args, wav_seg_points, midi_seg_points])

    pool = multiprocessing.Pool(processes=8)
    utt_id_all_list = pool.map(segment_func_new, midi_wav_args_list)
    utt_id_all_list_flatten = [utt_id for utt_id_list in utt_id_all_list for utt_id in utt_id_list]

    # write "wav_segments.scp" & "text_segments 0"
    utt_id_all_list_flatten.sort()
    for utt_id in utt_id_all_list_flatten:
        wav_segment_dir = os.path.join(args.wav_segments_dir, utt_id + '.wav')
        midi_segment_dir = os.path.join(args.text_segments_dir, utt_id + '.npz')
        wav_segments_list.append('{} {}\n'.format(utt_id, wav_segment_dir))
        midi_segments_list.append('{} {}\n'.format(utt_id, midi_segment_dir))
        if args.use_perf:
            perf_segment_dir = os.path.join(args.perf_segments_dir, utt_id + '.npz')
            perf_segments_list.append('{} {}\n'.format(utt_id, perf_segment_dir))
            

    wav_seg_dir = os.path.join(
        os.path.dirname(args.wav_dir), 'wav_segments.scp'
    )
    midi_set_dir = os.path.join(
        os.path.dirname(args.text_dir), 'text_segments.scp'
    )
    
    if args.use_perf:
        perf_set_dir = os.path.join(
        os.path.dirname(args.perf_dir), 'perf_segments.scp'
        )

    f_wav_segments = open(wav_seg_dir, 'w')
    f_midi_segments = open(midi_set_dir, 'w')
    if args.use_perf:
        f_perf_segments = open(perf_set_dir, 'w')

    if args.use_perf:
        for wav_seg, midi_seg, perf_seg in zip(wav_segments_list, midi_segments_list, perf_segments_list):
            f_wav_segments.write(wav_seg)
            f_midi_segments.write(midi_seg)
            f_perf_segments.write(perf_seg)
        f_wav_segments.close()
        f_midi_segments.close()
        f_perf_segments.close()
    else:
        for wav_seg, midi_seg in zip(wav_segments_list, midi_segments_list):
            f_wav_segments.write(wav_seg)
            f_midi_segments.write(midi_seg)
        f_wav_segments.close()
        f_midi_segments.close()


if __name__ == "__main__":
    main()