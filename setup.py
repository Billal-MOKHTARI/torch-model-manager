from setuptools import setup, find_packages

setup(
    name='torch-model-manager',
    version='1.0.0.dev1',
    description='A package for managing PyTorch models',
    author='Billal MOKHTARI',
    author_email='mokhtaribillal1@gmail.com',
    packages=find_packages(),
    url = 'https://github.com/Billal-MOKHTARI/torch-model-manager',
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