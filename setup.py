from setuptools import find_namespace_packages, setup

__version__ = "0.4.0rc3"
url = "https://github.com/ntt123/pax"

install_requires = ["jax", "jmp", "opax"]
setup_requires = []
tests_requires = [
    "chex",
    "dm-haiku",
    "fire",
    "mypy",
    "opax",
    "pytest",
    "tensorflow_datasets",
    "tensorflow",
    "tqdm",
]

setup(
    name="pax3",
    version=__version__,
    description="A stateful pytree library for training neural networks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thông Nguyễn",
    url=url,
    keywords=[
        "deep-learning",
        "jax",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    packages=find_namespace_packages(exclude=["examples", "tests"]),
    extras_require={"test": tests_requires},
    python_requires=">=3.7",
)
