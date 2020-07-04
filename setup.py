import os
from setuptools import find_packages, setup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(BASE_DIR, 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name="nlu",
    description="Natural language understanding project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    author="Anton Ivanov",
    email="a.i.ivanov.sv@gmail.com",
    classifiers=["Programming Language :: Python :: 3.6"],
)
