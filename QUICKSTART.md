# Quick Start Guide
**Sentiment Analysis Project - ISY503 Assessment 3**

This guide will help you get the project running quickly.

---

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **pip (Python package manager)**
   ```bash
   pip --version
   ```

3. **Git** (for version control)
   ```bash
   git --version
   ```

---

## 🚀 Quick Setup (5 Steps)

### Step 1: Download the Project

Option A - If you have Git:
```bash
git clone [YOUR_GITHUB_URL]
cd sentiment-analysis-project
```

Option B - Without Git:
- Download the ZIP file from GitHub
- Extract it
- Open terminal/command prompt in the extracted folder

---

### Step 2: Download the Dataset

1. Visit: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
2. Download the Amazon product reviews dataset
3. Extract the files
4. Place positive reviews in: `data/positive/`
5. Place negative reviews in: `data/negative/`

**Note:** If you can't download the dataset, the training script includes sample data for demonstration.

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (for neural networks)
- Flask (for web interface)
- NumPy, Pandas (for data processing)
- And other required packages

**Note:** This may take 5-10 minutes depending on your internet speed.

---

### Step 4: Train the Model

Navigate to the `src` folder and run:

```bash
cd src
python train_model.py
```

**What this does:**
1. Loads and preprocesses the data (or uses sample data if dataset not available)
2. Builds the neural network
3. Trains the model (this may take 10-30 minutes depending on your computer)
4. Evaluates performance
5. Saves the trained model to `models/`

**Expected output:**
```
Training Accuracy: ~90%
Validation Accuracy: ~85%
Test Accuracy: ~85%
```

---

### Step 5: Run the Web Interface

Navigate to the `web_interface` folder and run:

```bash
cd ../web_interface
python app.py
```

Then open your web browser and go to:
```
http://localhost:5000
```

You should see the sentiment analysis interface!

---

## 🧪 Testing the System

### Test with Sample Reviews

1. Open http://localhost:5000 in your browser

2. Try these example reviews:

**Positive Review:**
```
This product is absolutely amazing! The quality exceeded my expectations 
and it arrived quickly. I've been using it daily and couldn't be happier. 
Highly recommend to anyone looking for a reliable product!
```

**Negative Review:**
```
Terrible experience. The product broke after just one week of use. 
Customer service was unhelpful and refused to provide a refund. 
Complete waste of money. Do not buy this product!
```

**Neutral Review:**
```
It's okay, nothing special. Does what it's supposed to do but there 
are probably better options available for the same price.
```

---

## 🔧 Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```

Make sure you're in the project root directory.

---

### Problem: "Model not found" error when running web interface

**Solution:**
Train the model first:
```bash
cd src
python train_model.py
```

---

### Problem: TensorFlow installation fails

**Solution:**
Try installing a specific version:
```bash
pip install tensorflow==2.13.0
```

For Apple Silicon Macs:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

---

### Problem: Training takes too long

**Solution:**
Reduce the dataset size or epochs in `train_model.py`:
```python
CONFIG = {
    'epochs': 5,  # Reduce from 10 to 5
    ...
}
```

---

### Problem: Out of memory error

**Solution:**
Reduce batch size in `train_model.py`:
```python
CONFIG = {
    'batch_size': 32,  # Reduce from 64 to 32
    ...
}
```

---

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── data/                          # Put your dataset here
│   ├── positive/                  # Positive reviews
│   └── negative/                  # Negative reviews
│
├── models/                        # Trained models saved here
│   ├── sentiment_model.h5        # Main model
│   └── preprocessor.pkl          # Preprocessor config
│
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data cleaning
│   ├── model_architecture.py     # Neural network
│   ├── train_model.py           # Training script (RUN THIS FIRST)
│   └── prediction.py            # Prediction functions
│
├── web_interface/               # Web application
│   ├── app.py                   # Flask server (RUN THIS SECOND)
│   ├── templates/
│   │   └── index.html          # Web page
│   └── static/
│       └── style.css           # Styling
│
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
└── ethical_considerations.md    # Ethics discussion
```

---

## 🎯 Common Commands

### Training
```bash
cd src
python train_model.py
```

### Running Web Interface
```bash
cd web_interface
python app.py
```

### Interactive Prediction Demo
```bash
cd src
python prediction.py demo
```

### Initialize Git Repository
```bash
bash setup_git.sh
```

---

## 🌐 Setting Up GitHub

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name: `sentiment-analysis-project`
   - Don't initialize with README

2. Run the Git setup script:
   ```bash
   bash setup_git.sh
   ```

3. Link to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-project.git
   git branch -M main
   git push -u origin main
   ```

4. Your repository URL:
   ```
   https://github.com/YOUR_USERNAME/sentiment-analysis-project
   ```

---

## 📊 Expected Results

After training, you should see:

**Model Performance:**
- Training Accuracy: 85-95%
- Validation Accuracy: 80-90%
- Test Accuracy: 80-90%

**Web Interface:**
- Fast predictions (<1 second)
- Confidence scores displayed
- Clean, professional interface

---

## 🆘 Need Help?

1. Check the `README.md` for detailed documentation
2. Review `ethical_considerations.md` for project context
3. Look at code comments for implementation details
4. Check GitHub Issues for known problems
5. Ask your team members or facilitator

---

## ✅ Checklist

Before submission, ensure:

- [ ] Model trained successfully
- [ ] Web interface working
- [ ] Code pushed to GitHub
- [ ] Individual report completed
- [ ] Presentation prepared
- [ ] Team members listed in all documents
- [ ] GitHub repository link ready to submit

---


