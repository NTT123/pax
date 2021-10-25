"""Setup PAX3 package."""

from setuptools import find_namespace_packages, setup


def _get_version():
    with open("pax/__init__.py", encoding="utf-8") as file:
        for line in file:
            if line.startswith("__version__"):
                _globals = {}
                exec(line, _globals)  # pylint: disable=exec-used
                return _globals["__version__"]
        raise ValueError("`__version__` not defined in `pax/__init__.py`")


__version__ = _get_version()
URL = "https://github.com/ntt123/pax"

install_requires = ["jax>=0.2.21", "jmp>=0.0.2"]
setup_requires = []
tests_requires = [
    "chex",
    "dm-haiku",
    "fire",
    "opax",
    "pytest",
    "pytype",
    "tqdm",
]

setup(
    name="pax3",
    version=__version__,
    description="A stateful pytree library for training neural networks.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Thông Nguyễn",
    url=URL,
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
    include_package_data=True,
    zip_safe=False,
)
