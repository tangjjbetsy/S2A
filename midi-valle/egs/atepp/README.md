# ATEPP

## Install deps
```
pip install librosa==0.8.1
```

## Prepare Dataset
The midi files and audio files were cut into segments first and then the midis were tokenized with the tokenzier used in `score-to-performance` project ([here](https://github.com/tangjjbetsy/RHEPP-Transformer-S2P)).
```
cd egs/atepp

# Those stages are very time-consuming
python atepp.py
bash prepare.sh --stage -2 --stop-stage 3
```

The `data.tar` contains the resulted data files.