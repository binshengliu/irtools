from setuptools import setup, find_packages

setup(
    name='irtools',
    version='1.0.3',
    description='Utilities for IR research',
    author='Binsheng Liu',
    author_email='liubinsheng@gmail.com',
    scripts=[
        "scripts/each_server.sh", "irtools/perplexity.py",
        "irtools/mypyrouge.py", "irtools/indri.py", "scripts/trec_eval.py",
        "scripts/wtl.py", "scripts/cleanit.py", "scripts/bertit.py",
        "scripts/spacit.py", "scripts/binarize.py", "scripts/sample.py",
        "scripts/eval_run.py"
    ],
    packages=find_packages(exclude=['docs', 'tests', 'scripts']),
    include_package_data=True,
    install_requires=[
        'unidecode', 'tqdm', 'scipy', 'numpy', 'more-itertools', 'pandas',
        'ftfy', 'lxml', 'GPUtil'
    ],
)
