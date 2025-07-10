"""
Preprocessing utilities for fake review detection.
"""
import re
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
nltk.download('punkt', download_dir=nltk_data_path, quiet=False)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=False)

# Now import the resources
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)
    except LookupError as e:
        print(f"NLTK resource error: {e}")
        print("Falling back to simple tokenization without stopword removal")
        # Simple fallback if NLTK resources aren't available
        return text

def preprocess_text(text, remove_stops=True):
    """
    Full preprocessing pipeline.
    
    Args:
        text (str): Raw text input
        remove_stops (bool): Whether to remove stopwords
        
    Returns:
        str: Fully preprocessed text
    """
    text = clean_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    return text
