#!/bin/sh
#SBATCH --job-name=fine_tune_midi2wav
#SBATCH --out="train.out.txt"
#SBATCH --time=3-0
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:4

# module load cuda11.1

# print the node name
/bin/hostname

# load Pytorch dependency through conda
# NAME is the name of your conda environment 
./home/smg/v-jtbetsy/projects/midi-to-audio/tools/activate_python.sh

# run the Pytorch code
./run.sh --ngpu 4

# inference
./run.sh --stage 7 --stop_stage 7 --skip_data_prep true --ngpu 4 --tts_task gan_mta

#concat wav
python pyscripts/concat_waveforms.py exp/tts_finetune_joint_transformer_hifigan_atepp_cutpedal_raw_proll/decode_train.total_count.best/test/wav