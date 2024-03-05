#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_4

# Activate virtual enviroment 
#source ./assignment4/bin/activate 

# install hdbscan for BERTopic
#sudo apt-get update
#sudo apt-get install python3-dev

# requirements
#pip3 install transformers
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Deactivate the virtual environment.
#deactivate