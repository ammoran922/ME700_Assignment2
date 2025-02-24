# Assignment 2

## Table of Contents
1. Installation Instructions
2. Direct Stiffness Method

## 1. Installation Instructions
To install this package, please begin by setting up a conda environment:
```
mamba create --name assignment2 python=3.9.12
```

Once the environment has been created, activate it:
```
mamba activate assignment2
```

Double check that python is version 3.9.12 in the environment:
```
python --version
```

Ensure that pip is using the most up to date version of setuptools:
```
pip install --upgrade pip setuptools wheel
```

Create an editable install of the code (note: you must be in the same directory as the .toml file):
```
python -m pip install -e .
```

Test that the code is working with pytest:
```
pytest -v --cov=bisectionmethod --cov-report term-missing
```
Code coverage should be 100%.

## 2. Direct Stiffness Method
Direct Stiffness Method Implementation

### Example 1
TBD
