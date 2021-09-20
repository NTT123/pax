# Denoising Diffusion Model

We transcribe the PyTorch model at https://github.com/lucidrains/denoising-diffusion-pytorch

The implementation is almost identical to the PyTorch version. However, the `GaussianDiffusion.p_sample` method has to return the updated `self` to make it available outside the jitted method.