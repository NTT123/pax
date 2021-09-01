from setuptools import find_packages, setup

__version__ = "0.2.3"
url = "https://github.com/ntt123/pax"

install_requires = ["dm-haiku", "jax", "optax"]
setup_requires = []
tests_require = ["tensorflow", "tensorflow_datasets", "tqdm", "chex", "pytest"]

setup(
    name="pax",
    version=__version__,
    description="A Pytree <3 Jax Framework.",
    author="Thông Nguyễn",
    url=url,
    keywords=[
        "deep-learning",
        "jax",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(exclude=["examples", "tests"]),
    extras_require={"test": tests_require},
    python_requires=">=3.6",
)
