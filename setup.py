from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="roi_processing",
    version="0.1.0",
    author="",
    author_email="",
    description="Filtering and post-processing of s2p ROIs and traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ROI_processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
) 