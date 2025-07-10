"""
Model utilities for fake review detection.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_pretrained_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Load a pretrained transformer model for sequence classification.
    
    Args:
        model_name (str): Name of the pretrained model
        num_labels (int): Number of output classes
        
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return model, tokenizer

def predict(model, tokenizer, text, device="cpu"):
    """
    Make a prediction for a single text input.
    
    Args:
        model: The model to use for prediction
        tokenizer: The tokenizer for the model
        text (str): Input text
        device (str): Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Prepare the model
    model.to(device)
    model.eval()
    
    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
    # Get the predicted class and confidence
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence
