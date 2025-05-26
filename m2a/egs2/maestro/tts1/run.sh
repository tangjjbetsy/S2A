#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

fs=24000
n_fft=32768
# n_fft=8192
n_shift=288
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=train
valid_set=validation
test_sets=test
datadir=data
stage=2
stop_stage=6
ngpu=1
tts_task=gan_mta # gan_mta or score_to_audio
# datadir=data_spk_lang #score-to-audio

# train_config=conf/train.yaml
train_config=conf/tuning/finetune_joint_transformer_hifigan_atepp.yaml
# train_config=conf/tuning/finetune_joint_transformer_hifigan_atepp_score_with_sid_lid.yaml #score-to-audio

inference_config=conf/decode.yaml
inference_model=train.total_count.best.pth

g2p=g2p_en_no_space # Include no word separator

log "$0 $*"
. utils/parse_options.sh "$@"


./midi_to_wav.sh \
    --lang en \
    --feats_type raw \
    --feats_extract midifbank \
    --audio_format wav \
    --fs "${fs}" \
    --fmin 5 \
    --fmax 12000 \
    --n_mels 128 \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type proll \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_model "${inference_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --datadir "${datadir}" \
    --srctexts "${datadir}/${train_set}/text" \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --tts_task ${tts_task} \
    --gpu_inference true \
    --skip_data_prep false \
    # --tag finetune_joint_transformer_hifigan_atepp \ # midi-to-audio fine-tune

    ### Score-to-Audio #####
    # --use_sid true \ 
    # --use_lid true \
    # --tag finetune_joint_transformer_hifigan_atepp_s2a_with_sid_lid \ # score-to-audio fine-tune \
    # --dumpdir dump_spk_lang \
    # --tts_stats_dir exp/tts_stats_sid_lid
    ${opts} "$@"
