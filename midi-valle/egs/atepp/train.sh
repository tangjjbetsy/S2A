#!/bin/sh
#SBATCH --job-name=valle-ar
#SBATCH --out="exp/slurm.txt"
#SBATCH --time=3-0
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:4

/bin/hostname

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /home/smg/v-jtbetsy/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate valle

module load cuda11.7

## --train-stage 0 AR&NAR
## --train-stage 1 AR
## --train-stage 2 NAR

python3 bin/trainer.py --num-buckets 6 --save-every-n 10000 --valid-interval 10000 --train-stage 2 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false   \
    --decoder-dim 1024 --nhead 8 --num-decoder-layers 8 --prefix-mode 2   \
    --base-lr 0.05 --warmup-steps 200 --average-period 0    \
    --num-epochs 20 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4    \
    --exp-dir exp/valle-pre-2 --dataset atepp --max-duration 100 --world-size 4