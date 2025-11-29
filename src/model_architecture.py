"""
Neural Network Model for Sentiment Analysis
Author: [Your Team Name]
Date: November 2024

This module defines the neural network architecture using:
- Embedding layer for word representations
- LSTM (Long Short-Term Memory) layers for sequence processing
- Dense layers for classification
- Dropout for regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


class SentimentModel:
    """
    Neural Network model for binary sentiment classification
    """
    
    def __init__(self, vocab_size, max_length, embedding_dim=128):
        """
        Initialize model configuration
        
        Args:
            vocab_size (int): Size of vocabulary
            max_length (int): Maximum sequence length
            embedding_dim (int): Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build the neural network architecture
        
        Architecture:
        1. Embedding Layer: Converts word indices to dense vectors
        2. LSTM Layer 1: Processes sequences (returns sequences)
        3. LSTM Layer 2: Processes sequences (returns last output)
        4. Dropout: Prevents overfitting
        5. Dense Layer: Final classification layer with sigmoid activation
        
        Returns:
            keras.Model: Compiled model
        """
        print("\n=== Building Neural Network ===")
        
        model = models.Sequential([
            # Embedding layer: converts word indices to dense vectors
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding_layer'
            ),
            
            # First LSTM layer: processes sequences with 64 units
            # return_sequences=True: passes all outputs to next LSTM layer
            layers.LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_layer_1'
            ),
            
            # Second LSTM layer: processes sequences with 32 units
            # return_sequences=False: only returns final output
            layers.LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_layer_2'
            ),
            
            # Dropout layer: randomly drops 50% of neurons during training
            # Helps prevent overfitting
            layers.Dropout(0.5, name='dropout_layer'),
            
            # Dense layer: final classification layer
            # 1 unit with sigmoid activation for binary classification
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Print model summary
        print("\n=== Model Architecture ===")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        """
        Train the neural network
        
        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training labels
            X_val (np.array): Validation sequences
            y_val (np.array): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history
        """
        print("\n=== Training Model ===")
        
        # Callbacks for training
        callbacks = [
            # Early stopping: stops training if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint: saves best model during training
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        print("\n=== Training Complete ===")
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test (np.array): Test sequences
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n=== Evaluating Model ===")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, predicted_labels, 
                                    target_names=['Negative', 'Positive']))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, predicted_labels)
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_labels': predicted_labels
        }
    
    def predict(self, sequences):
        """
        Make predictions on new sequences
        
        Args:
            sequences (np.array): Padded sequences
            
        Returns:
            np.array: Predictions (probabilities)
        """
        predictions = self.model.predict(sequences)
        return predictions
    
    def predict_sentiment(self, sequence):
        """
        Predict sentiment for a single sequence
        
        Args:
            sequence (np.array): Single padded sequence
            
        Returns:
            tuple: (sentiment_label, confidence)
        """
        # Reshape for prediction
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        
        # Determine sentiment
        if prediction > 0.5:
            sentiment = "Positive"
            confidence = prediction
        else:
            sentiment = "Negative"
            confidence = 1 - prediction
        
        return sentiment, confidence
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Alternative simpler model for faster training
class SimpleSentimentModel:
    """
    Simpler model for faster training/testing
    """
    
    def __init__(self, vocab_size, max_length, embedding_dim=64):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
    
    def build_model(self):
        """
        Build a simpler, faster model
        """
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\n=== Simple Model Built ===")
        model.summary()
        
        return model


# Example usage
if __name__ == "__main__":
    print("Sentiment Model Module")
    print("=" * 50)
    
    # Example configuration
    vocab_size = 10000
    max_length = 200
    
    # Build model
    sentiment_model = SentimentModel(vocab_size, max_length)
    model = sentiment_model.build_model()
    
    print("\nModel ready for training!")
