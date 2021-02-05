import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nested_cross_val",
    version="0.0.1",
    author="Nicolas Captier",
    author_email="nicolas.captier@curie.fr",
    description="Nested cross-validation in python (compatible with scikit-learn API)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncaptier/nested_cross_val",
    packages=setuptools.find_packages(),
    install_requires = [
        "dask-ml >= 1.7.0",
        "joblib >= 0.16.0",
        "matplotlib >= 3.2.2",
        "numpy >= 1.18.5",
        "pandas >= 1.0.5",
        "scikit-learn >= 0.23.1",
        "umap-learn == 0.4.6",        
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
