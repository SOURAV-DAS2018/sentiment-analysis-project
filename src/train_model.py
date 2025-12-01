"""
Complete Training Pipeline 
Group 10


This version includes ALL improvements:
- Uses ALL 4 product categories (Books, DVD, Electronics, Kitchen)
- Optimized model architecture
- Better training settings
- Expected accuracy: 80-88%
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from data_preprocessing import DataPreprocessor

# ULTIMATE CONFIGURATION - ALL OPTIMIZED!
CONFIG = {
    'data_dir': '../data',
    # NOW USING ALL 4 CATEGORIES!
    'categories': ['books', 'dvd', 'electronics', 'kitchen_&_housewares'],
    'models_dir': '../models',
    'vocab_size': 8000,  # OPTIMIZED - not too small, not too large
    'max_length': 150,  # OPTIMIZED - balanced length
    'min_review_length': 10,
    'embedding_dim': 100,  # INCREASED for better word representations
    'epochs': 25,  # MORE epochs for better learning
    'batch_size': 32,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}


def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(CONFIG['models_dir'], exist_ok=True)
    print("✓ Directories created/verified")


def plot_training_history(history, save_path='../models/training_history.png'):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()


def load_all_categories(preprocessor, categories):
    """
    Load data from ALL product categories
    """
    all_data = []
    
    print("\n=== Loading Data from ALL Categories ===")
    
    for category in categories:
        category_path = os.path.join('../data', category)
        
        if os.path.exists(category_path):
            print(f"\nLoading {category}...")
            
            # Load positive reviews
            positive_file = os.path.join(category_path, 'positive.review')
            if os.path.exists(positive_file):
                positive_reviews = preprocessor.parse_review_file(positive_file)
                # Take up to 1250 reviews per category
                positive_reviews = positive_reviews[:1250]
                for review in positive_reviews:
                    all_data.append((review, 1))
                print(f"  ✓ Loaded {len(positive_reviews)} positive reviews")
            
            # Load negative reviews
            negative_file = os.path.join(category_path, 'negative.review')
            if os.path.exists(negative_file):
                negative_reviews = preprocessor.parse_review_file(negative_file)
                # Take up to 1250 reviews per category
                negative_reviews = negative_reviews[:1250]
                for review in negative_reviews:
                    all_data.append((review, 0))
                print(f"  ✓ Loaded {len(negative_reviews)} negative reviews")
        else:
            print(f"  ⚠️  Category {category} not found, skipping...")
    
    print(f"\n✓ Total reviews from all categories: {len(all_data)}")
    return all_data


def build_optimized_model(vocab_size, max_length, embedding_dim):
    """
    Build OPTIMIZED model with better architecture
    """
    from tensorflow.keras import models, layers
    from tensorflow.keras.regularizers import l2
    
    print("\n=== Building OPTIMIZED Neural Network ===")
    
    model = models.Sequential([
        # Embedding layer with more dimensions
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # Spatial dropout for embedding layer
        layers.SpatialDropout1D(0.2),
        
        # Global average pooling
        layers.GlobalAveragePooling1D(),
        
        # First dense layer - larger
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Second dense layer
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Third dense layer
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use a lower learning rate for better convergence
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== Model Architecture ===")
    model.summary()
    
    return model


def main():
    """
    Main training pipeline - ULTIMATE VERSION
    """
    print("="*70)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("ULTIMATE VERSION - ALL IMPROVEMENTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Create directories
    print("STEP 1: Setting up directories...")
    create_directories()
    print()
    
    # Step 2: Initialize preprocessor
    print("STEP 2: Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        vocab_size=CONFIG['vocab_size'],
        max_length=CONFIG['max_length'],
        min_review_length=CONFIG['min_review_length']
    )
    print("✓ Preprocessor initialized")
    print()
    
    # Step 3: Load data from ALL categories
    print("STEP 3: Loading dataset from ALL categories...")
    print(f"Categories: {', '.join(CONFIG['categories'])}")
    
    data = load_all_categories(preprocessor, CONFIG['categories'])
    
    if len(data) < 100:
        print("\n⚠️  WARNING: Very few reviews loaded!")
        print("Please check that the data folders exist")
    
    print()
    
    # Step 4: Preprocess data
    print("STEP 4: Preprocessing data...")
    X, y = preprocessor.prepare_data(data)
    print("✓ Data preprocessing complete")
    print()
    
    # Step 5: Split data
    print("STEP 5: Splitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        test_ratio=CONFIG['test_ratio']
    )
    print("✓ Data split complete")
    print()
    
    # Step 6: Save preprocessor
    print("STEP 6: Saving preprocessor...")
    preprocessor.save_preprocessor(f"{CONFIG['models_dir']}/preprocessor.pkl")
    print()
    
    # Step 7: Build optimized model
    print("STEP 7: Building neural network...")
    model = build_optimized_model(
        vocab_size=CONFIG['vocab_size'],
        max_length=CONFIG['max_length'],
        embedding_dim=CONFIG['embedding_dim']
    )
    print("✓ Model built successfully")
    print()
    
    # Step 8: Train model
    print("STEP 8: Training model...")
    print(f"Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}")
    print("This will take 20-30 minutes - please be patient!")
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,  # More patience for better training
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f"{CONFIG['models_dir']}/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print("\n=== Training Model ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Training Complete ===")
    print("✓ Training complete")
    print()
    
    # Step 9: Plot training history
    print("STEP 9: Plotting training history...")
    plot_training_history(history)
    print()
    
    # Step 10: Evaluate model
    print("STEP 10: Evaluating model on test set...")
    print("\n=== Evaluating Model ===")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    # Calculate metrics
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
    
    # Calculate per-class accuracy
    neg_accuracy = cm[0,0] / (cm[0,0] + cm[0,1]) * 100
    pos_accuracy = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    print(f"\nNegative Reviews Accuracy: {neg_accuracy:.2f}%")
    print(f"Positive Reviews Accuracy: {pos_accuracy:.2f}%")
    print()
    
    # Step 11: Save final model
    print("STEP 11: Saving trained model...")
    model.save(f"{CONFIG['models_dir']}/sentiment_model.h5")
    print(f"Model saved to {CONFIG['models_dir']}/sentiment_model.h5")
    print()
    
    # Step 12: Test with sample predictions
    print("STEP 12: Testing with sample predictions...")
    test_reviews = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality. Complete waste of money.",
        "Good value for the price. Recommended.",
        "Very disappointed. Does not work at all.",
        "Excellent purchase! Exceeded expectations!",
        "Broke after one use. Would not recommend."
    ]
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS:")
    print("="*70)
    
    for review in test_reviews:
        # Preprocess
        cleaned = preprocessor.clean_text(review)
        sequence = preprocessor.text_to_sequence(cleaned)
        padded = preprocessor.pad_sequence(sequence).reshape(1, -1)
        
        # Predict
        prediction = model.predict(padded, verbose=0)[0][0]
        
        if prediction > 0.5:
            sentiment = "Positive"
            confidence = prediction
        else:
            sentiment = "Negative"
            confidence = 1 - prediction
        
        print(f"\nReview: '{review}'")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2%})")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel saved to: {CONFIG['models_dir']}/sentiment_model.h5")
    print(f"Preprocessor saved to: {CONFIG['models_dir']}/preprocessor.pkl")
    print(f"\n🏆 Final Test Accuracy: {accuracy:.2%}")
    
   
    
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
