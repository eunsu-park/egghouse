from setuptools import setup, find_packages

setup(
    name="egghouse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas"        
    ],
    python_requires=">=3.9"
)