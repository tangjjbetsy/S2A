# Objective Evaluation for Synthesised Midi Performances

For reproducing the results in the paper, please follow these steps:
1. Calculate features for each system using `00_get.py`.

```bash
feat=chroma # or midispec
json_path=exp_sys.json #include all systems in the json file
python 00_get.py --feat $feat --data-json $json_path
```
2. Compute the distance of the features between any two systems using `01_compute_dis.py`.

```bash
feat=chroma # or midispec
con_sys=groundtruth # or sys1, sys2, etc.
exp_sys=baseline # or sys1, sys2, etc.
python 01_compute_dis.py --feat $feat --con_sys $con_sys --exp_sys $exp_sys
```

