import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="people_flow",
    version="0.0.1",
    author="KMASAHIRO",
    description="simulating people flow and infer the parameters from actual people flow data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KMASAHIRO/people_flow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
)
