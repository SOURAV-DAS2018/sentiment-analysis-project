"""
Prediction Module for Sentiment Analysis
Author: [Your Team Name]
Date: November 2024

This module provides functions for making predictions on new text:
- Load trained model and preprocessor
- Process new review text
- Return sentiment prediction with confidence
"""

import numpy as np
import pickle
from tensorflow import keras


class SentimentPredictor:
    """
    Handles prediction on new text using trained model
    """
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize predictor by loading model and preprocessor
        
        Args:
            model_path (str): Path to saved model (.h5 file)
            preprocessor_path (str): Path to saved preprocessor (.pkl file)
        """
        print("Loading model and preprocessor...")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load preprocessor configuration
        with open(preprocessor_path, 'rb') as f:
            config = pickle.load(f)
        
        self.vocab_size = config['vocab_size']
        self.max_length = config['max_length']
        self.word_to_index = config['word_to_index']
        self.index_to_word = config['index_to_word']
        print(f"✓ Preprocessor loaded from {preprocessor_path}")
        print(f"  Vocabulary size: {len(self.word_to_index)}")
        print(f"  Max sequence length: {self.max_length}")
    
    def clean_text(self, text):
        """
        Clean input text (same as preprocessing)
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned text
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of integers
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of word indices
        """
        words = text.split()
        sequence = [self.word_to_index.get(word, 1) for word in words]  # 1 = UNK
        return sequence
    
    def pad_sequence(self, sequence):
        """
        Pad or truncate sequence to max_length
        
        Args:
            sequence (list): List of word indices
            
        Returns:
            np.array: Padded/truncated sequence
        """
        if len(sequence) > self.max_length:
            return np.array(sequence[:self.max_length])
        else:
            padded = np.zeros(self.max_length, dtype=int)
            padded[:len(sequence)] = sequence
            return padded
    
    def predict(self, text, return_details=False):
        """
        Predict sentiment for input text
        
        Args:
            text (str): Review text to analyze
            return_details (bool): If True, return detailed information
            
        Returns:
            tuple: (sentiment, confidence) or dict if return_details=True
        """
        # Preprocess text
        cleaned_text = self.clean_text(text)
        sequence = self.text_to_sequence(cleaned_text)
        padded_sequence = self.pad_sequence(sequence)
        
        # Reshape for prediction
        input_data = padded_sequence.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(input_data, verbose=0)[0][0]
        
        # Determine sentiment
        if prediction > 0.5:
            sentiment = "Positive"
            confidence = prediction
        else:
            sentiment = "Negative"
            confidence = 1 - prediction
        
        if return_details:
            return {
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'confidence': float(confidence),
                'raw_prediction': float(prediction),
                'sequence_length': len(sequence)
            }
        else:
            return sentiment, confidence
    
    def predict_batch(self, texts):
        """
        Predict sentiments for multiple texts
        
        Args:
            texts (list): List of review texts
            
        Returns:
            list: List of tuples (sentiment, confidence)
        """
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append((sentiment, confidence))
        return results


def interactive_prediction_demo(model_path='../models/sentiment_model.h5',
                                preprocessor_path='../models/preprocessor.pkl'):
    """
    Interactive demo for testing predictions
    
    Args:
        model_path (str): Path to saved model
        preprocessor_path (str): Path to saved preprocessor
    """
    print("="*70)
    print("SENTIMENT ANALYSIS - INTERACTIVE DEMO")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = SentimentPredictor(model_path, preprocessor_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running: python train_model.py")
        return
    
    print("\n✓ Predictor ready!")
    print("\nType 'quit' or 'exit' to stop")
    print("-"*70)
    
    while True:
        # Get user input
        print("\nEnter a product review:")
        text = input("> ")
        
        # Check for exit
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Skip empty input
        if not text.strip():
            continue
        
        # Make prediction
        try:
            result = predictor.predict(text, return_details=True)
            
            # Display results
            print("\n" + "="*70)
            print("PREDICTION RESULT:")
            print("="*70)
            print(f"Original Text: {result['text']}")
            print(f"Cleaned Text:  {result['cleaned_text']}")
            print(f"\n🎯 Sentiment:   {result['sentiment']}")
            print(f"📊 Confidence:  {result['confidence']:.2%}")
            print(f"📈 Raw Score:   {result['raw_prediction']:.4f}")
            print(f"📝 Words Used:  {result['sequence_length']}")
            print("="*70)
            
        except Exception as e:
            print(f"Error making prediction: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Run interactive demo
        interactive_prediction_demo()
    else:
        # Example predictions
        print("Sentiment Prediction Module")
        print("="*70)
        
        # Check if model exists
        import os
        model_path = '../models/sentiment_model.h5'
        preprocessor_path = '../models/preprocessor.pkl'
        
        if not os.path.exists(model_path):
            print("⚠️  Model not found!")
            print("Please train the model first by running:")
            print("  python train_model.py")
            print("\nOr run interactive demo with:")
            print("  python prediction.py demo")
        else:
            # Initialize predictor
            predictor = SentimentPredictor(model_path, preprocessor_path)
            
            # Test examples
            test_reviews = [
                "This product is absolutely amazing! Best purchase ever!",
                "Terrible quality. Don't waste your money on this.",
                "It's okay, nothing special but works as expected.",
                "Love it! Exactly what I needed. Highly recommend!",
                "Worst product I've ever bought. Complete garbage."
            ]
            
            print("\nTesting sample reviews:")
            print("-"*70)
            
            for review in test_reviews:
                sentiment, confidence = predictor.predict(review)
                print(f"\nReview: '{review}'")
                print(f"→ {sentiment} (Confidence: {confidence:.2%})")
