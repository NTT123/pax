import jax

from model import GaussianDiffusion, UNet

model = UNet(dim=64, dim_mults=(1, 2, 4, 8))

diffusion = GaussianDiffusion(
    model, image_size=128, timesteps=1000, loss_type="l1"  # number of steps  # L1 or L2
)

print(diffusion.summary())

training_images = jax.random.normal(jax.random.PRNGKey(42), (2, 128, 128, 3))
loss = diffusion(training_images)

# sampled_images = diffusion.sample(batch_size = 4)
