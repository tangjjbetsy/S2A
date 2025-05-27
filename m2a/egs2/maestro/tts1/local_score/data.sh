#!/usr/bin/env bash

set -e
set -u
# set -x
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2

sample_rate=16000
num_segment_frame=800

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# ROOT=`pwd | sed 's%\(.*/REPO\)/.*%\1%'`
# maestro_root="${ROOT}/../${tgt_folder}/maestro/Google_maestro-v2.0.0"
# maestro_root="${MAESTRO}/maestro/MIDI-filterbank"
db_root="/home/smg/v-jtbetsy/DATA"
tgt_folder="data_score"
use_sid=True
use_lid=True
use_perf=False

train_set="train"
valid_set="validation"
test_sets="test"

segmentation_points_dir='/home/smg/v-jtbetsy/DATA/ATEPP-s2a/segmentation_points.npy'

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # generate downloads/maestro-v3.0.0
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Split to Subsets"
    # split data into ${tgt_folder}/{train/validation/test} dataset
    # generate wav.scp & text in each dset
    # wav.scp: wav_id wav_dir
    # text: wav_id midi_dir
    [ -e ${tgt_folder} ] && rm -r ${tgt_folder}
    # python local/data_parse.py "${db_root}" data
    python local/data_parse.py ${db_root} ${tgt_folder} ${use_sid} ${use_lid} ${use_perf}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Generate Segmentation for both midi & audio"
    # make segments (wav_segments.scp, wav_segments, text_segments.scp, text_segments)
    # wav_segments.scp (eg: segid ${tgt_folder}/{train/val/text}/wav_segments/{segid}.wav)
    # text_segments.scp (eg: segid ${tgt_folder}/{train/val/text}/text_segments/{segid}.npz)

    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        python local/data_segments.py --wav_dir ${tgt_folder}/"${dset}"/wav.scp \
            --wav_segments_dir ${tgt_folder}/"${dset}"/wav_segments \
            --text_dir ${tgt_folder}/"${dset}"/text \
            --text_segments_dir ${tgt_folder}/"${dset}"/text_segments \
            --perf_dir ${tgt_folder}/"${dset}"/perf \
            --perf_segments_dir ${tgt_folder}/"${dset}"/perf_segments \
            --sample_rate ${sample_rate} \
            --num_segment_frame ${num_segment_frame} \
            --segmentation_points_dir ${segmentation_points_dir} \
            --use_perf True
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generate utt2spk, utt2lang & spk2utt, lang2utt"
    # make utt2spk & spk2utt
    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        utt2spk=${tgt_folder}/"${dset}"/utt2spk
        spk2utt=${tgt_folder}/"${dset}"/spk2utt
        utt2lang=${tgt_folder}/"${dset}"/utt2lang
        lang2utt=${tgt_folder}/"${dset}"/lang2utt
        [ -e ${utt2spk} ] && mv ${utt2spk} ${tgt_folder}/"${dset}"/utt2spk_original
        [ -e ${spk2utt} ] && mv ${spk2utt} ${tgt_folder}/"${dset}"/spk2utt_original
        [ -e ${utt2lang} ] && mv ${utt2lang} ${tgt_folder}/"${dset}"/utt2lang_original
        [ -e ${lang2utt} ] && mv ${lang2utt} ${tgt_folder}/"${dset}"/lang2utt_original
        
        mv ${tgt_folder}/"${dset}"/wav.scp ${tgt_folder}/"${dset}"/wav_original.scp
        mv ${tgt_folder}/"${dset}"/text ${tgt_folder}/"${dset}"/text_original
        mv ${tgt_folder}/"${dset}"/wav_segments.scp ${tgt_folder}/"${dset}"/wav.scp
        mv ${tgt_folder}/"${dset}"/text_segments.scp ${tgt_folder}/"${dset}"/text

        python local/generate_utt2spk.py ${tgt_folder}/"${dset}"/wav.scp ${tgt_folder}/"${dset}"/utt2spk ${tgt_folder}/"${dset}"/utt2lang
        utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
        utils/utt2lang_to_lang2utt.pl ${utt2lang} > ${lang2utt}

        [ -e ${utt2spk} ] && rm ${tgt_folder}/"${dset}"/utt2spk_original
        [ -e ${spk2utt} ] && rm ${tgt_folder}/"${dset}"/spk2utt_original
        [ -e ${utt2lang} ] && rm ${tgt_folder}/"${dset}"/utt2lang_original
        [ -e ${lang2utt} ] && rm ${tgt_folder}/"${dset}"/lang2utt_original
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
