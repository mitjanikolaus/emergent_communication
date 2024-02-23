# Emergent Communication

This is the repo for the ICLR 2024 paper "Emergent Communication with Conversational Repair" by Mitja Nikolaus.

### Python environment

A python environment can be setup by using the environment files [environment.yml](environment.yml) or
[environment_cpu.yml](environment_cpu.yml) (for CPU-only).

## Basic signaling game
A basic signaling game experiment can be started by running:
```
python train.py --sender-layer-norm --receiver-layer-norm
```

For setups with noise and feedback channel simply add the corresponding command line args:
```
python train.py --sender-layer-norm --receiver-layer-norm --noise 0.5 --feedback
```

## GuessWhat signaling game

For a GuessWhat signaling game, first run the feature extraction script:
```
python extract_guesswhat_features.py
```

Afterwards a training run can be started:
```
python train.py --guesswhat --sender-layer-norm --receiver-layer-norm
```

## Further configuration options

All config options are displayed when running simply:
```
python train.py -h
```

## Result plotting

Results were plotted using the Jupyter notebook [print_result_plots.ipynb](print_result_plots.ipynb).
