#!/usr/bin/env python
"""
Setup script to download required NLTK resources.
Run this once before using the notebook.
"""
import nltk
import ssl
import os

print("Setting up NLTK resources...")

# Fix SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

print("NLTK setup complete!")
