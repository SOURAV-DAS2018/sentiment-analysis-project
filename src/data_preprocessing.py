"""
Data Preprocessing Module for Sentiment Analysis - FIXED VERSION
Author: [Your Team Name]
Date: November 2024

This module handles:
1. Loading positive and negative reviews from Amazon .review files
2. Text cleaning (removing punctuation, special characters)
3. Text tokenization and encoding
4. Label encoding
5. Outlier removal (very short reviews)
6. Padding/truncating sequences
7. Train/validation/test split
"""

import os
import re
import numpy as np
from collections import Counter
import pickle


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for sentiment analysis
    """
    
    def __init__(self, vocab_size=10000, max_length=200, min_review_length=10):
        """
        Initialize preprocessor with configuration
        
        Args:
            vocab_size (int): Maximum vocabulary size (most common words)
            max_length (int): Maximum sequence length for padding/truncating
            min_review_length (int): Minimum words in review (outlier removal)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_review_length = min_review_length
        self.word_to_index = {}
        self.index_to_word = {}
        
    def clean_text(self, text):
        """
        Clean review text by:
        - Converting to lowercase
        - Removing URLs
        - Removing HTML tags
        - Removing special characters and punctuation
        - Removing extra whitespace
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and punctuation (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def parse_review_file(self, filepath):
        """
        Parse Amazon .review file which contains XML-style reviews
        
        Args:
            filepath (str): Path to .review file
            
        Returns:
            list: List of review texts
        """
        reviews = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Split by <review> tags
            review_blocks = content.split('<review>')
            
            for block in review_blocks:
                if '</review>' in block:
                    # Extract the review text between <review_text> tags
                    match = re.search(r'<review_text>(.*?)</review_text>', block, re.DOTALL)
                    if match:
                        review_text = match.group(1).strip()
                        if review_text:
                            reviews.append(review_text)
            
            print(f"  Extracted {len(reviews)} reviews from {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
        
        return reviews
    
    def load_data_from_files(self, positive_dir, negative_dir):
        """
        Load reviews from positive.review and negative.review files
        
        Args:
            positive_dir (str): Directory containing positive.review file
            negative_dir (str): Directory containing negative.review file
            
        Returns:
            list: List of tuples (review_text, label)
        """
        data = []
        
        print("Loading positive reviews...")
        positive_file = os.path.join(positive_dir, 'positive.review')
        if os.path.exists(positive_file):
            positive_reviews = self.parse_review_file(positive_file)
            # Limit to 5000 reviews for faster training
            positive_reviews = positive_reviews[:5000]
            for review in positive_reviews:
                data.append((review, 1))  # 1 = positive
            print(f"  Total positive reviews loaded: {len(positive_reviews)}")
        else:
            print(f"  WARNING: {positive_file} not found!")
        
        print("Loading negative reviews...")
        negative_file = os.path.join(negative_dir, 'negative.review')
        if os.path.exists(negative_file):
            negative_reviews = self.parse_review_file(negative_file)
            # Limit to 5000 reviews for faster training
            negative_reviews = negative_reviews[:5000]
            for review in negative_reviews:
                data.append((review, 0))  # 0 = negative
            print(f"  Total negative reviews loaded: {len(negative_reviews)}")
        else:
            print(f"  WARNING: {negative_file} not found!")
        
        print(f"Total reviews loaded: {len(data)}")
        return data
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from text data
        Creates word-to-index and index-to-word mappings
        
        Args:
            texts (list): List of cleaned text strings
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 for PAD and UNK
        
        # Create mappings
        # 0 = PAD (padding), 1 = UNK (unknown word)
        self.word_to_index = {'PAD': 0, 'UNK': 1}
        self.index_to_word = {0: 'PAD', 1: 'UNK'}
        
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_index)}")
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of integers using vocabulary
        
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
            # Truncate
            return np.array(sequence[:self.max_length])
        else:
            # Pad with zeros
            padded = np.zeros(self.max_length, dtype=int)
            padded[:len(sequence)] = sequence
            return padded
    
    def remove_outliers(self, data):
        """
        Remove reviews that are too short (likely not useful)
        
        Args:
            data (list): List of tuples (text, label)
            
        Returns:
            list: Filtered data
        """
        filtered_data = []
        removed_count = 0
        
        for text, label in data:
            word_count = len(text.split())
            if word_count >= self.min_review_length:
                filtered_data.append((text, label))
            else:
                removed_count += 1
        
        print(f"Removed {removed_count} reviews with < {self.min_review_length} words")
        print(f"Remaining reviews: {len(filtered_data)}")
        
        return filtered_data
    
    def prepare_data(self, data):
        """
        Complete data preparation pipeline:
        1. Clean text
        2. Remove outliers
        3. Build vocabulary
        4. Convert to sequences
        5. Pad sequences
        
        Args:
            data (list): List of tuples (raw_text, label)
            
        Returns:
            tuple: (X, y) where X is padded sequences, y is labels
        """
        print("\n=== Data Preparation Pipeline ===")
        
        # Step 1: Clean all texts
        print("\n1. Cleaning text...")
        cleaned_data = [(self.clean_text(text), label) for text, label in data]
        
        # Step 2: Remove outliers
        print("\n2. Removing outliers...")
        cleaned_data = self.remove_outliers(cleaned_data)
        
        # Step 3: Build vocabulary
        print("\n3. Building vocabulary...")
        texts = [text for text, label in cleaned_data]
        self.build_vocabulary(texts)
        
        # Step 4: Convert to sequences
        print("\n4. Converting text to sequences...")
        sequences = [self.text_to_sequence(text) for text in texts]
        
        # Step 5: Pad sequences
        print("\n5. Padding sequences...")
        X = np.array([self.pad_sequence(seq) for seq in sequences])
        y = np.array([label for text, label in cleaned_data])
        
        print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train, validation, and test sets
        
        Args:
            X (np.array): Input sequences
            y (np.array): Labels
            train_ratio (float): Proportion for training
            val_ratio (float): Proportion for validation
            test_ratio (float): Proportion for testing
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Shuffle data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Calculate split indices
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        print("\n=== Data Split ===")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath):
        """
        Save preprocessor configuration for later use
        
        Args:
            filepath (str): Path to save pickle file
        """
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'min_review_length': self.min_review_length,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word
        }
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load preprocessor configuration
        
        Args:
            filepath (str): Path to pickle file
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.vocab_size = config['vocab_size']
        self.max_length = config['max_length']
        self.min_review_length = config['min_review_length']
        self.word_to_index = config['word_to_index']
        self.index_to_word = config['index_to_word']
        
        print(f"Preprocessor loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(vocab_size=10000, max_length=200, min_review_length=10)
    
    # Test with electronics data
    data = preprocessor.load_data_from_files('../data/electronics', '../data/electronics')
    
    if len(data) > 0:
        print(f"\nSuccessfully loaded {len(data)} reviews!")
        print(f"First review sample: {data[0][0][:100]}...")
    else:
        print("\nNo data loaded - check file paths!")
