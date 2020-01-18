from setuptools import setup, find_packages

setup(
    name='irtools',
    version='1.0',
    description='Utilities for IR research',
    author='Binsheng Liu',
    author_email='liubinsheng@gmail.com',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'unidecode', 'tqdm', 'scipy', 'numpy', 'more_itertools', 'pandas'
    ],
)
