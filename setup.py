# setup.py
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='fact-checker-pkg',
    version='0.1.0',
    description='A package for scoring facts using KGE Models with PyKEEN.',
    author='Luke Friedrichs',
    author_email='lukef@mail.uni-paderborn.de',
    url='https://github.com/yourusername/fact-checker',
    packages=find_packages(where="."),  
    install_requires=[
        'pykeen',
        'rdflib',
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
			"factCheck=fokg_mini_project.scripts.run:main"
        ],
    },
    include_package_data=True,
)
