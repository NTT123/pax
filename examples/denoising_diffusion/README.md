## Denoising Diffusion Model

We transcribe the PyTorch model at https://github.com/lucidrains/denoising-diffusion-pytorch.

The implementation is almost identical to the PyTorch version. 
The difference is at how PAX manages random keys. PAX's version uses a `RngSeq` submodule to generates new random keys when needed.

To train model:

```sh
pip install -r requirements.txt
python3 train.py
```