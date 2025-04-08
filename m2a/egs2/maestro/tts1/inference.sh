#!/bin/sh
#SBATCH --job-name=inference_midi2wav
#SBATCH --out="inference.out.1.txt"
#SBATCH --time=1-0
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:4

# print the node name
/bin/hostname

# load Pytorch dependency through conda
# NAME is the name of your conda environment 
./home/smg/v-jtbetsy/projects/midi-to-audio/tools/activate_python.sh

#Load cuda version
# module load cuda11.1
# run the Pytorch code
./run.sh --stage 7 --stop_stage 7 --skip_data_prep true --ngpu 4 --tts_task gan_mta

#concat wav
python pyscripts/concat_waveforms.py exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll/decode_train.total_count.best/test/wav