#!/bin/bash

Virtual environment setup
pip install virtualenv
echo 'Sudo password is required to install python3-venv'
sudo apt install python3-venv
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up project structure
bash utils/scripts/projectStructure.sh

# Run the application
python app.py