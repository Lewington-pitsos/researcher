import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="researcher",
    version="0.0.21",
    author="Louka Ewington-Pitsos",
    author_email="lewingtonpitsos@gmail.com",
    description="A tool for recording the results of data science experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy', 'numpy', 'matplotlib'],
    python_requires='>=3.6',
    test_suite='nose.collector',
    tests_require=['nose'],
)