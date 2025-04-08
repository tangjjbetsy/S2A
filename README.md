# Score-to-Audio: Towards an Integrated Approach for Expressive Piano Performance Synthesis from Music Scores
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