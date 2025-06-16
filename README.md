# Towards an Integrated Approach for Expressive Piano Performance Synthesis from Music Scores
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J16U55C-uBYgMDasUC7-Ku8zirzjFVQb?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2501.10222v1-b31b1b.svg)](https://arxiv.org/abs/2501.10222v1)
![Conference](https://img.shields.io/badge/Conference-ICASSP%202025-blue)

This repository contains the official implementation of our ICASSP 2025 [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10890623)

**"Towards an Integrated Approach for Expressive Piano Performance Synthesis from Music Scores"**

by Jingjing Tang, Erica Cooper, Xin Wang, Junichi Yamagishi, and György Fazekas.

## Project Structure

- **`m2a/`**  
  Contains the implementation for fine-tuning the MIDI-to-audio synthesis model using the ATEPP dataset, as well as the baseline model setup.

- **`m2m/`**  
  Includes the expressive performance rendering model, designed to generate expressive performance MIDI files from symbolic music scores.

- **`objective_eval/`**
  Scripts to run objective evaluation of synthesised MIDI Performances

## How to Use

Please refer to the `README.md` files inside each subdirectory (`m2a/` and `m2m/`) for detailed instructions on inference and generation of target MIDI or audio outputs.

As for reproducting objective evaluation results of the m2a model, please refer to the `objective_eval/README.md` file. For the m2m model, the matrix could be reproduced by running the evaluation script in the `m2m/` directory.

We also provide a colab notebook for quick testing of both models: [Colab Notebook](https://colab.research.google.com/drive/1J16U55C-uBYgMDasUC7-Ku8zirzjFVQb?usp=sharing).

## Dataset & Checkpoints
The dataset for training the `m2m` model and all the checkpoints could be downloaded from [Zenodo](https://zenodo.org/records/15524693). For the dataset used to finetune the `m2a` model, please contact the authors directly.

## Demo
You can listen to the demo samples on our [project page](https://tangjjbetsy.github.io/S2A/).

## Contact
Jingjing Tang: `jingjing.tang@qmul.ac.uk`

## License
The code is licensed under Apache License Version 2.0, following ESPnet. The pretrained model is licensed under the Creative Commons License: Attribution 4.0 International http://creativecommons.org/licenses/by/4.0/legalcode

## Acknowledgements
This work is supported by both the UKRI Centre for Doctoral Training in Artificial Intelligence and Music (grant number EP/S022694/1), and the National Institute of Informatics in Japan. J.Tang is a research student supported jointly by the China Scholarship Council [grant number 202008440382] and Queen Mary University of London. E. Cooper conducted this work while at NII, Japan and is currently employed by NICT, Japan.

## Reference
```
@INPROCEEDINGS{10890623,
  author={Tang, Jingjing and Cooper, Erica and Wang, Xin and Yamagishi, Junichi and Fazekas, György},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Towards An Integrated Approach for Expressive Piano Performance Synthesis from Music Scores}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10890623}}
```