module load cuda11.7

exp_dir=exp/valle

python3 bin/infer.py --output-dir infer/ \
    --checkpoint=${exp_dir}/best-valid-loss.pt \
    --text-prompts "/home/smg/v-jtbetsy/DATA/ATEPP-valle/prompt/05871_1_0_prompt_0s.npy" \
    --audio-prompts "/home/smg/v-jtbetsy/DATA/ATEPP-valle/prompt/05871_1_0_prompt_0s.wav" \
    --text "/home/smg/v-jtbetsy/DATA/ATEPP-valle/midi_seg/test/05871_1_0.npy|\
/home/smg/v-jtbetsy/DATA/ATEPP-valle/midi_seg/test/05871_1_1.npy|\
/home/smg/v-jtbetsy/DATA/ATEPP-valle/midi_seg/test/05871_1_2.npy|\
/home/smg/v-jtbetsy/DATA/ATEPP-valle/midi_seg/test/05871_1_4.npy|\
/home/smg/v-jtbetsy/DATA/ATEPP-valle/midi_seg/test/05871_1_5.npy"
    # --continual True


# --text 