
# Midi-to-Audio Model
The current implementation supports CUDA only, and cannot be run on CPU.

## Installation
1. Create conda enviroment:

```bash
conda create -n midi2wav python=3.9 -y
conda activate midi2wav
```

2. Set up Kaldi:
```bash
cd <midi2wav-root>/tools
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
ln -s <kaldi-root> .
```

3. Install ESPnet
`wget` is required for this step, please install it if you don't have it.

```bash
$ cd <midi2wav-root>/tools
$ make TH_VERSION=1.10.1 CUDA_VERSION=11.3 # Please select the versions that fits your CUDA and device.
```
Make sure the espnet version is `espnet==0.10`. 

4. Extra dependencies:
```bash
pretty_midi==0.2.10
wandb==0.12.9
protobuf==3.19.3
pandas==2.2.1
librosa==0.10.1
typeguard==2.13.3
scipy==1.10.0
```
*Please refer to the [MIDI-to-Audio repo](https://github.com/nii-yamagishilab/midi-to-audio/tree/main) instructions for more details about installation.*


## Training & Evaluation

### Prepare
1. Working Dictionary: `egs2/maestro/tts1/`
```bash
cd <midi2wav-root>/egs2/maestro/tts1/
```
2. Please [download](https://zenodo.org/records/15524693/files/m2a.zip) all the checkpoints for `m2a` and save them under `exp/` folder.

3. Download `train.loss.ave.pth` checkpoint pre-trained by the original [M2A](https://github.com/nii-yamagishilab/midi-to-audio/tree/main) implementation from [Zenodo](https://zenodo.org/records/7439325#.Y5pcAi8Rr0o) and save it under `exp/tts_finetune_joint_transformer_hifigan_raw_proll/` folder.

4.  To dowload the training datasets, please leave you email or contact `jingjing.tang@qmul.ac.uk` for the access.

5. Use the 
`conf/tuning/finetune_joint_transformer_hifigan_atepp.yaml` as the configuration to finetune the model. You could specify the pre-trained model path with `init_param` in the configuration file, currently set to `exp/tts_finetune_joint_transformer_hifigan_raw_proll/train.loss.ave.pth`.

### Run
```bash
# 1. Prepare the data segments
./local/data.sh --mode train --path_to_data DATA_DIR
# or
./local_score/data.sh # For training the baseline model

# 2. Process the data for training
train_config=conf/tuning/finetune_joint_transformer_hifigan_atepp.yaml
num_gpus=1 # Set the number of GPUs you want to use, set to 0 for CPU training
inference_model=train.total_count.best.pth

./run.sh --stage 2 --stop_stage 5 --train_config ${train_config} --ngpu ${num_gpus} --tts_task gan_mta

# 3. Train the model
./run.sh --stage 6 --stop_stage 6 --train_config ${train_config} --ngpu ${num_gpus} --tts_task gan_mta

# 4. Test the model
./run.sh --stage 7 --stop_stage 7 --train_config ${train_config} --inference_model ${inference_model} --ngpu ${num_gpus} --tts_task gan_mta
```

For training the baseline model, please use the `exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll/train.total_count.best.pth` as pre-trained model and use the `conf/tuning/finetune_joint_transformer_hifigan_atepp_score_with_sid_lid.yaml` as the configuration to finetune the model.

### Inference
Prepare the midis that you would like to synthesize under a folder, e.g., `PATH_TO_MIDI_FOLDER`, and run the following commands:

```bash
# 1. Prepare the data segments for inference
./local/data.sh --mode "inference" --inference_path PATH_TO_MIDI_FOLDER

# 2. Process the data for inference
./run.sh --stage 2 --stop_stage 2 --ngpu 1 --tts_task gan_mta

# 3. Create a `feats_type` file under `dump/raw/test` by
mkdir -p dump/raw/test
echo "raw" > dump/raw/test/feats_type
echo "'feats_type' file created with content 'raw' in dump/raw/test"

# 4. Run inference
train_config=conf/tuning/finetune_joint_transformer_hifigan_atepp.yaml
inference_model=train.total_count.best.pth
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --tts_task gan_mta --inference_model ${inference_model} --train_config ${train_config}

# 5. To concatenate the audio files, please run the the following command:
python pyscripts/concatenate_audio.py PATH_TO_SEGMENTS
```