"""
Helper script to download NLTK resources.
Run this before running the notebook if you encounter NLTK resource errors.
"""
import nltk
import os
import ssl

# Fix SSL certificate issues that can occur with NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK resources with explicit download path
nltk_data_path = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
print(f"Downloading NLTK resources to {nltk_data_path}")

# Force download of required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

print("NLTK resources downloaded successfully!")
