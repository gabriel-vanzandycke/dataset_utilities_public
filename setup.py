from setuptools import setup, find_packages

setup(
    name='viprior_dataset_utilities',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@synergysports.com",
    url="https://gitlab.com/deepsport/dataset_utilities",
    licence="LGPL",
    python_requires='>=3.6',
    description="",
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "opencv-python",
        "imageio",
        "calib3d>=2.2.0",
        "mlworkflow>=0.3.9",
        "shapely",
    ],
)
