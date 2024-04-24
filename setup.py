from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torch-model-manager',
    version='0.0.6.dev1',
    description='A package for managing PyTorch models',
    author='Billal MOKHTARI',
    author_email='mokhtaribillal1@gmail.com',
    packages=find_packages(exclude=['tests', 'docs', 'build.sh']),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/Billal-MOKHTARI/torch-model-manager',
    keywords=['PyTorch', 'Deep Learning', 'Machine Learning', 'High Level Programming'],
    install_requires=[
        'torch',
        'numpy',
        'torchvision',
        'torch'
        # Add any other dependencies here
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],

)