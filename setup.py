from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="egghouse",
    version="0.10.0",
    author="Eunsu Park",
    description="Utility library for solar physics research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eunsu-park/egghouse",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",

    # Core dependencies. The SDO / DEM / transfer stacks were promoted from
    # optional extras into core so a plain `pip install -e .` yields a fully
    # working install (JSOC download via drms, Level-1.5 prep + wavelength
    # response via aiapy/sunpy, CHIANTI temperature response via fiasco, HTTP
    # download via requests/bs4). Only genuinely niche stacks stay in extras.
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "astropy>=5.0",
        "sunpy>=4.0",
        "matplotlib>=3.5",
        "aiapy>=0.7",
        "drms>=0.7",
        "fiasco",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
    ],

    extras_require={
        # Database module (egghouse.database)
        "database": [
            "psycopg2-binary>=2.9.0",
            "pyyaml>=6.0",
        ],
        # SFTP support (for transfer module)
        "sftp": [
            "paramiko>=3.0.0",
        ],
        # Classical denoisers (egghouse.denoise)
        "denoise": [
            "scikit-image>=0.20",
            "PyWavelets>=1.4",
            "bm3d>=4.0",
        ],
        # OpenCV-backed image transform (egghouse.image.transforms, lazy import)
        "image": [
            "opencv-python>=4.5",
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
        # Backward-compat no-op aliases: the SDO / DEM / transfer / FITS stacks
        # were promoted into core (a plain install now includes them), so these
        # extras install nothing extra — kept only so existing `egghouse[sdo]`
        # style references keep resolving. Prefer a plain `pip install egghouse`.
        "sdo": [],
        "dem": [],
        "transfer": [],
        "fits": [],
        # All optional dependencies (beyond core)
        "all": [
            "psycopg2-binary>=2.9.0",
            "pyyaml>=6.0",
            "paramiko>=3.0.0",
            "scikit-image>=0.20",
            "PyWavelets>=1.4",
            "bm3d>=4.0",
            "opencv-python>=4.5",
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