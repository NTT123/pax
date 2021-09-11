## Introduction

This example is an implementation of [Lyra](https://github.com/google/lyra) WaveGRU network. There are two big differences:

* We predict the 8-bit mu-compressed waveform instead of the raw 16-bit waveform.
* The `UpsamplingNetwork` can look ahead into the future, hence it cannot be used in the online/streaming mode.


## Data preparation

We use `ffmpeg` and `sox` to do audio conversion and silence trimming.


To prepare audio clip:

    pip3 install -r requirements.txt
    bash prepare_data.sh
    
## Train WaveGRU

    python3 train.py # 1 hours on a Tesla T4
