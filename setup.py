from setuptools import setup, find_packages
setup(
    name='arcAGI2024',
    version='0.1.4',
    description='Your package description here',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where="src/main"),
    package_dir={'': 'src/main'},
    install_requires=[
        'torch',
        'tokenizers', # add other dependencies here
        'transformers',
        'typeguard',
        'numpy'
    ],
    python_requires='>=3.10', # adjust based on your requirements
)