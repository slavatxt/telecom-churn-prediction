from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="telecom-churn-prediction",
    version="0.1.0",
    author="Vyacheslav",
    author_email="your.email@example.com",
    description="ML project for predicting customer churn in telecom company",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slavatxt/telecom-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "joblib>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:main",
            "churn-predict=src.models.predict:main",
        ],
    },
)
