# Amazon Product Review Sentiment Analysis

## Project Overview
This project implements a Neural Network-based sentiment analysis system that classifies Amazon product reviews as either positive or negative.

## Team Members
- [Your Name] - [Student ID]
- [Team Member 2] - [Student ID]
- [Team Member 3] - [Student ID]
- [Team Member 4] - [Student ID]

## Project Structure
```
sentiment-analysis-project/
│
├── data/                          # Dataset folder (you need to download)
│   ├── positive/                  # Positive reviews
│   └── negative/                  # Negative reviews
│
├── models/                        # Saved trained models
│   └── sentiment_model.h5
│
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data cleaning and preparation
│   ├── model_training.py         # Neural network training
│   ├── model_evaluation.py       # Testing and evaluation
│   └── prediction.py             # Inference functions
│
├── web_interface/                # Web application
│   ├── app.py                    # Flask web server
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/
│       └── style.css            # Styling
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── ethical_considerations.md     # Ethics discussion

```

## Setup Instructions

### 1. Download Dataset
- Visit: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
- Download the Amazon product reviews dataset
- Extract and place in the `data/` folder

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python src/model_training.py
```

### 4. Run the Web Interface
```bash
python web_interface/app.py
```
Then open: http://localhost:5000

## Ethical Considerations
See `ethical_considerations.md` for detailed discussion of:
- Bias in sentiment classification
- Data privacy concerns
- Fair representation across product categories
- Transparency in AI decision-making

## Model Performance
- Training Accuracy: [To be filled]
- Validation Accuracy: [To be filled]
- Test Accuracy: [To be filled]

## References
- Dataset: Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
