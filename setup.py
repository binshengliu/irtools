from setuptools import setup

setup(
    name='ir_tools',
    version='1.0',
    description='Utilities for IR research',
    author='Binsheng Liu',
    author_email='liubinsheng@gmail.com',
    packages=['ir_tools'],
    install_requires=[
        'unidecode', 'tqdm', 'scipy', 'numpy', 'more_itertools', 'pandas'
    ],
)
