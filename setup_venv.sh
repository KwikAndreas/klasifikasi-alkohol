#!/bin/bash

echo "Setting up Python virtual environment for Wine Quality Classifier..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.6 or later."
    exit 1
fi

# Check if venv module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "The venv module is not available. Please install Python 3.6 or later."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Virtual environment setup complete!"
echo
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo
echo "To deactivate the virtual environment, run:"
echo "deactivate"
echo
