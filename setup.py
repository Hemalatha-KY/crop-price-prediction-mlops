from setuptools import setup, find_packages

setup(
    name="crop-price-prediction",
    version="1.0.0",
    description="MLOps project for crop price prediction using ML models",
    author="MLOps Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "flask>=3.0.0",
        "dvc>=3.48.0",
        "clearml>=1.15.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
