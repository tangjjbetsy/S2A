# Score-to-Audio
This repository contains the official implementation of our ICASSP 2025 [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10890623), **"Towards an Integrated Approach for Expressive Piano Performance Synthesis from Music Scores."**

## Project Structure

- **`m2a/`**  
  Contains the implementation for fine-tuning the MIDI-to-audio synthesis model using the ATEPP dataset, as well as the baseline model setup.

- **`m2m/`**  
  Includes the expressive performance rendering model, designed to generate expressive performance MIDI files from symbolic music scores.

## How to Use

Please refer to the `README.md` files inside each subdirectory (`m2a/` and `m2m/`) for detailed instructions on inference and generation of target MIDI or audio outputs.

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