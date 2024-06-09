#!/bin/sh
#SBATCH --job-name=s2p_bert_class
#SBATCH --out="logs/train_class_DA.txt"
#SBATCH --time=2-0
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:2

# /bin/hostname

# if [ -z "${PS1:-}" ]; then
#     PS1=__dummy__
# fi
# . /home/smg/v-jtbetsy/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate s2p

python src/train.py experiment=s2p_bert_class model.optimizer.lr=2e-5 task_name="s2p_bert_class_DA" model.dynamic_weights=False trainer.max_epochs=1000
