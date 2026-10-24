#!/bin/bash

# Define the Python version to install
PYTHON_VERSION="3.14.9"

# Install specified Python version
echo "Installing Python $PYTHON_VERSION if needed..."
if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    pyenv install $PYTHON_VERSION
else
    echo "Python $PYTHON_VERSION is already installed"
fi
pyenv local $PYTHON_VERSION

poetry install
