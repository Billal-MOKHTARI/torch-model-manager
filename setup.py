from setuptools import setup, find_packages

setup(
    name='torch-model-manager',
    version='1.0.0.dev1',
    description='A package for managing PyTorch models',
    author='Your Name',
    author_email='your@email.com',
    packages=find_packages(include=['core/*'],
                           exclude=['tests/*', 'docs/*']),
    url = 'https://github.com/Billal-MOKHTARI/torch-model-manager',
    long_description=open('README.md').read(),
    install_requires=[
        'torch',
        'numpy',
        'torchvision',
        'torch'
        # Add any other dependencies here
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],

)