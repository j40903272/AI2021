from setuptools import setup, find_packages

setup(
    name="GANBandit",
    version="1.0",
    author="Yunda Tsai",
    author_email="bb04902103@gmail.com",
    packages=find_packages('.'),
    python_requires='>=3.7',
    platforms=["any"],
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchvision"
    ]
)
