#!/bin/bash

# Define the Python version to install
PYTHON_VERSION="3.12.8"

# Install specified Python version
echo "Installing Python $PYTHON_VERSION..."
pyenv install $PYTHON_VERSION
pyenv local $PYTHON_VERSION

poetry install
