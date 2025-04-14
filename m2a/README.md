
# Midi-to-Audio Model

## Fine-Tune the [M2A](https://github.com/nii-yamagishilab/midi-to-audio/tree/main) for ATEPP Dataset

### Running

Working Dictionary: `egs2/maestro/tts1/`

We fine-tuned the `exp/tts_finetune_joint_transformer_hifigan_raw_proll/train.loss.ave.pth` model given by the original implementation.

To reproduce the result, please follow the original instruction given below to install the environment and use the 
`conf/tuning/finetune_joint_transformer_hifigan_atepp.yaml` as the configuration to finetune the model.

It would be recommended to prepare the data separately using `local/data.sh` first and then run from **stage 2 to 5** to prepare everything else. Then you could run **stage 6** for training, and **stage 7** for inferencing.

As for the training datasets, please leave you email or contact `jingjing.tang@qmul.ac.uk` for the access.

### Checkpoints
Please [download](https://drive.google.com/drive/folders/17lqEafXRI_mCUVnzjeq70NVqXD5VYTZI?usp=share_link) the checkpoints and save it under `exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll` folder.

## Baseline model
We fine-tuned the `exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll/train.total_count.best.pth` model given by the fisrt fine-tuning. We included the pianist identity as the speaker id, and ablum identity as the lang id. 

### Running
Working Dictionary: `egs2/maestro/tts1/`

To reproduce the result, please follow the original instruction given below to install the `m2a` environment and use the 
`conf/tuning/finetune_joint_transformer_hifigan_atepp_score_with_sid_lid.yaml` as the configuration to use the model.

It would be recommended to prepare the data separately using `local_score/data.sh` first and then run from **stage 2 to 5** to prepare everything else. Then you could run **stage 6** for training, and **stage 7** for inferencing.

------------------------------------
Please refer to the [MIDI-to-Audio repo](https://github.com/nii-yamagishilab/midi-to-audio/tree/main) instructions for installation.
------------------------------------
