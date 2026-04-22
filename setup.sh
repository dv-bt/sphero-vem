#!/bin/bash

# Define the Python version to install
PYTHON_VERSION="3.12.8"

# Install specified Python version
echo "Installing Python $PYTHON_VERSION if needed..."
if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    pyenv install $PYTHON_VERSION
else
    echo "Python $PYTHON_VERSION is already installed"
fi
pyenv local $PYTHON_VERSION

poetry install
