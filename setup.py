from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="egghouse",
    version="0.7.0",
    author="Eunsu Park",
    description="Utility library for solar physics research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eunsu-park/egghouse",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",

    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],

    extras_require={
        # Database module
        "database": [
            "psycopg2-binary>=2.9.0",
            "pyyaml>=6.0",
        ],
        # Transfer module (HTTP downloads)
        "transfer": [
            "requests>=2.25.0",
            "beautifulsoup4>=4.9.0",
        ],
        # SFTP support (for transfer module)
        "sftp": [
            "paramiko>=3.0.0",
        ],
        # FITS I/O (astropy-based functions)
        "fits": [
            "astropy>=5.0",
        ],
        # SDO processing (full functionality)
        "sdo": [
            "astropy>=5.0",
            "sunpy>=4.0",
        ],
        # DEM analysis (temperature response functions)
        "dem": [
            "aiapy>=0.7",
            "astropy>=5.0",
            "sunpy>=4.0",
        ],
        # Config module (YAML support)
        "config": [
            "pyyaml>=6.0",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        # All optional dependencies
        "all": [
            "psycopg2-binary>=2.9.0",
            "pyyaml>=6.0",
            "requests>=2.25.0",
            "beautifulsoup4>=4.9.0",
            "paramiko>=3.0.0",
            "astropy>=5.0",
            "sunpy>=4.0",
            "aiapy>=0.7",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    keywords="solar physics, SDO, AIA, HMI, astronomy, image processing",
)