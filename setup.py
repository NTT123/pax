from setuptools import setup

__version__ = "0.0.1"
url = "https://github.com/ntt123/pax"

install_requires = [
    "dm-haiku",
    "einops",
    "jax",
    "jaxlib",
    "optax",
]
setup_requires = []
tests_require = []

setup(
    name="pax",
    version=__version__,
    description="A Pytree Jax Framework.",
    author="ntt123",
    url=url,
    keywords=[
        "deep-learning",
        "jax",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=["pax"],
    python_requires=">=3.6",
)
