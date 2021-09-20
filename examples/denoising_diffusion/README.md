# Denoising Diffusion Model

We transcribe the PyTorch model at https://github.com/lucidrains/denoising-diffusion-pytorch.

The implementation is almost identical to the PyTorch version. 
The difference is at how Pax manages random keys. Pax's version uses a `RngSeq` submodule to generates new random keys when needed.

To train model:

```sh
pip3 install -r requirements.txt
python3 train.py
```