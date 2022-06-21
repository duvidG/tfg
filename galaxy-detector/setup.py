"""Setup for the SKA Science Data Challenge Scoring package."""

import setuptools

with open("README.md") as f:
    README = f.read()

setuptools.setup(
    author="SKA Organisation",
    name="ska_sdc",
    license="BSD 3-clause",
    description="A package providing tools for the SKA Science Data Challenges.",
    version="v2.0.0",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/ska-telescope/sdc/ska-sdc",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18",
        "pandas>=0.25",
        "scikit-learn>=0.22",
        "astropy>=4.0",
    ],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    package_data={"": ["data/beam_info/PB_I_14.log", "data/conf/sdc2/config.ini"]},
)
