from setuptools import find_packages, setup

__version__ = "0.2.11rc"
url = "https://github.com/ntt123/pax"

install_requires = [
    "jax",
    "jmp",
]
setup_requires = []
tests_require = [
    "chex",
    "dm-haiku",
    "fire",
    "mypy",
    "opax @ git+https://github.com/ntt123/opax",
    "pytest",
    "tensorflow_datasets",
    "tensorflow",
    "tqdm",
]

setup(
    name="pax",
    version=__version__,
    description="A stateful pytree library for training neural networks.",
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
