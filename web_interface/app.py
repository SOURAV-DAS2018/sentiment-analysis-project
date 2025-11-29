"""
Flask Web Application for Sentiment Analysis
Author: [Your Team Name]
Date: November 2024

This creates a simple web interface where users can:
- Enter a product review
- Click a button to analyze sentiment
- See the prediction result (Positive/Negative) with confidence
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import SentimentPredictor

# Initialize Flask app
app = Flask(__name__)

# Initialize predictor (will be loaded when app starts)
predictor = None

def load_model():
    """
    Load the trained model and preprocessor
    """
    global predictor
    
    model_path = os.path.join('..', 'models', 'sentiment_model.h5')
    preprocessor_path = os.path.join('..', 'models', 'preprocessor.pkl')
    
    try:
        predictor = SentimentPredictor(model_path, preprocessor_path)
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.route('/')
def home():
    """
    Home page with sentiment analysis form
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze sentiment of submitted review
    
    Returns:
        JSON response with sentiment and confidence
    """
    try:
        # Get review text from form
        data = request.get_json()
        review_text = data.get('review', '')
        
        # Validate input
        if not review_text or not review_text.strip():
            return jsonify({
                'error': 'Please enter a review to analyze'
            }), 400
        
        # Check if model is loaded
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Make prediction
        result = predictor.predict(review_text, return_details=True)
        
        # Return result
        return jsonify({
            'success': True,
            'sentiment': result['sentiment'],
            'confidence': round(result['confidence'] * 100, 2),
            'raw_prediction': round(result['raw_prediction'], 4)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error analyzing review: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'message': 'Service is running'
    })


if __name__ == '__main__':
    print("="*70)
    print("SENTIMENT ANALYSIS WEB APPLICATION")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    if load_model():
        print("\n✓ Application ready!")
        print("\nOpen your browser and go to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("="*70)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model")
        print("Please train the model first by running:")
        print("  python src/train_model.py")
        print("="*70)
