from setuptools import find_packages, setup


setup(
    name="gameoflife",
    version="0.3.0",
    description="Modernized Conway's Game of Life with sparse, dense, numba, and torch backends",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.8",
        "numpy>=1.26",
        "numba>=0.59",
    ],
    extras_require={
        "torch": [
            "torch>=2.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "gameoflife=gameoflife.cli:main",
        ]
    },
    url="https://github.com/amaynez/GameOfLife",
    project_urls={
        "Upstream": "https://github.com/amaynez/GameOfLife",
    },
)
