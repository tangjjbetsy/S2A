
# Midi-to-Audio Model

## Installation

1. Create conda enviroment:

```bash
conda create -n midi2wav python=3.9 -y
conda activate midi2wav
```

2. You don't have to set up kaldi for this

3. Install ESPnet
`wget` is required for this step, please install it if you don't have it.

```bash
$ cd <midi2wav-root>/tools
$ make TH_VERSION=1.10.1 CUDA_VERSION=11.3
```
Make sure the espnet version is `espnet==0.10` and `matplotlib` needs to be installed separately depending on the operation system.

1. Extra dependencies:
```
pretty_midi==0.2.10
wandb==0.12.9
protobuf==3.19.3
pandas==2.2.1
librosa==0.10.1
```
*Please refer to the [MIDI-to-Audio repo](https://github.com/nii-yamagishilab/midi-to-audio/tree/main) instructions for more details about installation.*

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

## Inference
To inference the audio of one or several midi files, please change the `mode` in `local/data.sh` to `"inference":
```
mode="inference"
inference_path=PATH_TO_MIDI_FOLDER
```
Then, run `local_data.sh` to prepare the segments and meta. 

Run the `egs2/maestro/tts1/run.sh` from ``stage 2 to 3`` by
```
./run.sh --stage 2 --stop_stage 3 --ngpu ${num_gpu} --tts_task gan_mta
```

Create a `feats_type` file under `dump/raw/test` by
```
mkdir -p dump/raw/test
echo "raw" > dump/raw/test/feats_type
echo "'feats_type' file created with content 'raw' in dump/raw/test"
```

Then, run the `egs2/maestro/tts1/run.sh` to inference the audio by
```
./run.sh --stage 7 --stop_stage 7 --ngpu ${num_gpu} --tts_task gan_mta
```

To concatenate the audio files, please run the the following command:
```
# Wav segments is usually saved to `exp/EXP_NAME/decode_train.total_count.best/test/wav`
python pyscripts/concatenate_audio.py PATH_TO_SEGMENTS
```