#!/usr/bin/env bash
# Create virtual environment
python3 -m venv assignment1_env

#Install packages
python -m spacy download en_core_web_md
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
