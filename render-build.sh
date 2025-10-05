#!/bin/bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Download spaCy model (needed for NLP)
python -m spacy download en_core_web_sm