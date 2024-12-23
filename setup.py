from setuptools import setup, find_packages

setup(
    name="fokg_mini_project",
    version="0.1.0",
    description="A mini project for knowledge graph fact veraccity checking",
    author="Luke Friedrichs",
    author_email="lukef@mail.uni-paderborn.de",
    python_requires=">=3.10.16",
    install_requires=[
        "torch>=2.5.1",  
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "test_train=scripts.run:main",  
        ],
    },
)
