from setuptools import setup, find_packages
print("PACKAGES", find_packages(where="src/main"))
setup(
    name='arcAGI2024',
    version='0.4.0',
    description='Support mechanism for the arc agi project',
    author='Chris',
    author_email='your.email@example.com',
    packages=find_packages(where="src/main"),
    package_dir={'': 'src/main'},
    install_requires=[
        'torch',
        'tokenizers', # add other dependencies here
        'transformers',
        'typeguard',
        'tqdm',
        'numpy',
        'pandas'
    ],
    python_requires='>=3.10', # adjust based on your requirements
)