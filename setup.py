"""Setup script for neural_trading package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="neural_trading",
    version="1.0.0",
    author="Neural Trading Team",
    author_email="team@neuraltrading.ai",
    description="TDD-based deep learning cryptocurrency trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/another-worlds/neural_trading",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pylint>=2.17.5",
            "ipython>=8.14.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "pytest-asyncio>=0.21.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-trading-train=src.training.cli:main",
            "neural-trading-infer=src.inference.cli:main",
            "neural-trading-backtest=src.backtesting.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)
