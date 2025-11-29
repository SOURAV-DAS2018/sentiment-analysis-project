# 🎥 Video Demonstration Script
**For Recording Your Presentation Demo**

---

## 📹 Pre-Recording Checklist

Before you start recording:

✅ Model is trained and working  
✅ Web interface tested  
✅ Browser cleared of personal bookmarks/tabs  
✅ Desktop cleaned (hide personal files)  
✅ Test microphone and screen recording software  
✅ Close unnecessary applications  
✅ Have example reviews ready to test  

---

## 🎬 Recording Script (10-15 minutes)

### PART 1: Introduction (30 seconds)

**[Show title slide or desktop]**

"Hello, we are [Team Name] and today we'll demonstrate our sentiment analysis project for ISY503. This system uses neural networks to automatically classify Amazon product reviews as positive or negative."

---

### PART 2: Project Overview (1 minute)

**[Show README or architecture diagram]**

"Our project consists of four main components:

1. Data preprocessing - which cleans and prepares review text
2. A neural network using LSTM architecture - which learns patterns in the text
3. A training pipeline - which trains the model on 10,000 Amazon reviews
4. A web interface - which makes the system accessible to users

The system achieves over 85% accuracy in classifying sentiment."

---

### PART 3: Code Structure (2 minutes)

**[Show file explorer with project structure]**

"Let me show you the code organization. 

[Navigate through folders]

In the 'src' folder, we have:
- data_preprocessing.py - handles text cleaning and tokenization
- model_architecture.py - defines our LSTM neural network
- train_model.py - orchestrates the training process
- prediction.py - makes predictions on new reviews

In the 'web_interface' folder:
- app.py - Flask server that handles HTTP requests
- templates/index.html - the user interface
- static/style.css - professional styling

The 'models' folder stores:
- Our trained model weights
- The preprocessor configuration
- Training history plots"

---

### PART 4: Training Process (2 minutes)

**[Open terminal/command prompt]**

"Now let me show you how we trained the model. I'll navigate to the src folder and run the training script."

**[Type commands]**
```bash
cd src
python train_model.py
```

**[While it's running or show pre-recorded training output]**

"As you can see, the training process:
1. Loads and preprocesses 10,000 reviews
2. Builds the neural network with 500,000 parameters
3. Trains for 10 epochs with early stopping
4. Validates performance on a separate dataset
5. Saves the best model

Our final results show:
- Training accuracy: [X]%
- Validation accuracy: [Y]%
- Test accuracy: [Z]%

This indicates good performance without overfitting."

---

### PART 5: Web Interface Demo (3-4 minutes)

**[Open terminal/command prompt]**

"Now let's start the web interface."

**[Type commands]**
```bash
cd web_interface
python app.py
```

**[Show terminal output confirming server started]**

"The Flask server is now running on localhost port 5000."

**[Open browser to http://localhost:5000]**

"Here's our web interface. It has:
- A clean, professional design with a gradient header
- A text area for entering reviews
- An analyze button to trigger sentiment analysis
- Example reviews users can try
- Clear display of results with confidence scores"

---

### PART 6: Live Testing (3-4 minutes)

**[Test Example 1: Strong Positive]**

"Let me test with a clearly positive review."

**[Type or paste]**
```
This product is absolutely amazing! The quality exceeded my expectations 
and it arrived quickly. I've been using it daily and couldn't be happier. 
Highly recommend to anyone looking for a reliable product!
```

**[Click "Analyze Sentiment"]**

"As you can see, the system correctly identifies this as a POSITIVE review with [X]% confidence. The high confidence indicates the model is very certain about this classification."

---

**[Test Example 2: Strong Negative]**

"Now let's try a clearly negative review."

**[Type or paste]**
```
Terrible experience. The product broke after just one week of use. 
Customer service was unhelpful and refused to provide a refund. 
Complete waste of money. Do not buy this product!
```

**[Click "Analyze Sentiment"]**

"The system correctly identifies this as NEGATIVE with [X]% confidence."

---

**[Test Example 3: Neutral/Borderline]**

"Let's test with a more neutral review to see how the system handles ambiguity."

**[Type or paste]**
```
It's okay, nothing special. Does what it's supposed to do but there 
are probably better options available for the same price. Shipping 
was average.
```

**[Click "Analyze Sentiment"]**

"Interesting - the system classifies this as [POSITIVE/NEGATIVE] with [X]% confidence. Notice the confidence is lower here because the sentiment is less clear. This is exactly what we want - the system should be less certain when the review is ambiguous."

---

**[Test Example 4: User Input]**

"Finally, let me try an example that wasn't in our training data."

**[Type something original]**
```
Fantastic gadget! Works perfectly and looks great. Five stars!
```

**[Click "Analyze Sentiment"]**

"The system handles new reviews well, correctly identifying this as POSITIVE."

---

### PART 7: Code Walkthrough (2-3 minutes)

**[Open code editor]**

"Let me briefly show you some key parts of the code."

**[Open model_architecture.py]**

"Here's our neural network architecture. We're using:
- An Embedding layer that converts words to 128-dimensional vectors
- Two LSTM layers with 64 and 32 units
- Dropout for regularization
- A final Dense layer with sigmoid activation for binary classification

The LSTM layers are particularly good for sequential data like text because they can remember context from earlier in the review."

**[Open data_preprocessing.py]**

"Our preprocessing pipeline includes:
- Text cleaning to remove noise
- Tokenization to convert words to numbers
- Padding to ensure all sequences are the same length
- Train/validation/test splitting

This ensures the neural network receives properly formatted input."

---

### PART 8: Ethical Considerations (2 minutes)

**[Show ethical_considerations.md or slides]**

"An important part of this project was considering the ethical implications. We identified several key concerns:

1. **Bias in training data** - The dataset may not represent all demographics equally, which could lead to unfair classifications

2. **Privacy** - Reviews might contain personal information, so we used only anonymized public data

3. **Fairness** - Misclassifications could harm businesses or mislead consumers, so we provide confidence scores and recommend human oversight

4. **Transparency** - Neural networks can be 'black boxes,' so we documented our architecture clearly and provide explainable metrics

5. **Commercial impact** - Automated sentiment analysis could significantly affect businesses, so we emphasize the importance of not relying solely on automated decisions

These considerations guided our implementation and are documented in detail in our ethical considerations document."

---

### PART 9: Technical Challenges (1 minute)

**[Can show slides or speak to camera]**

"We encountered several challenges during development:

1. **Overfitting** - Initially our model performed well on training data but poorly on validation data. We solved this by adding dropout layers and implementing early stopping.

2. **Data imbalance** - We needed to ensure equal representation of positive and negative reviews.

3. **Sarcasm detection** - The model struggles with sarcastic reviews like 'Oh great, another broken product.' This is a known limitation we've documented.

4. **Training time** - We optimized our architecture and batch size to balance accuracy with training speed."

---

### PART 10: Results Summary (1 minute)

**[Show results slide or training graphs]**

"Our final results show:

**Performance:**
- Test Accuracy: [X]%
- Precision: [X]%
- Recall: [X]%
- F1-Score: [X]%

**Strengths:**
- High accuracy on clear positive/negative reviews
- Fast prediction time (<1 second)
- User-friendly web interface
- Well-documented codebase

**Limitations:**
- Struggles with sarcasm and mixed sentiments
- Limited to English language
- May not generalize to domains outside Amazon reviews"

---

### PART 11: Future Improvements (30 seconds)

"Future enhancements could include:
- Multi-class classification for star ratings (1-5)
- Attention mechanisms to highlight important words
- Multi-lingual support
- Aspect-based sentiment analysis for specific features
- Real-time learning from new reviews"

---

### PART 12: Conclusion (30 seconds)

**[Return to title slide or face camera]**

"To summarize, we've successfully developed an end-to-end sentiment analysis system that:
- Achieves over 85% accuracy
- Provides a user-friendly web interface
- Carefully considers ethical implications
- Demonstrates practical application of AI in natural language processing

Our complete code and documentation are available on GitHub at [URL].

Thank you for watching. We're happy to answer any questions!"

---

## 🎬 Recording Tips

### Technical Setup
- **Screen Recording Software:**
  - Windows: OBS Studio, Camtasia, or built-in Xbox Game Bar
  - Mac: QuickTime Player or ScreenFlow
  - Online: Loom or Screencast-O-Matic

- **Microphone:**
  - Test audio levels before recording
  - Speak clearly and at moderate pace
  - Reduce background noise

- **Video Settings:**
  - 1920x1080 resolution (Full HD)
  - 30 FPS
  - MP4 format
  - H.264 codec

### Presentation Tips
- **Pacing:** Speak at 120-150 words per minute
- **Pauses:** Pause briefly after showing something new
- **Cursor:** Move cursor deliberately to guide viewer's eye
- **Zoom:** Zoom in on important code sections
- **Transitions:** Use smooth transitions between topics

### Editing
- Trim dead space at beginning/end
- Add title card (team name, project name)
- Add captions if time permits
- Check audio sync with video
- Export in high quality

---

## 📝 Pre-Recording Preparation

### Day Before Recording
1. ✅ Run through entire script once
2. ✅ Test all code and demos
3. ✅ Prepare example reviews
4. ✅ Clean desktop and browser
5. ✅ Update all documentation
6. ✅ Charge laptop fully

### Morning of Recording
1. ✅ Test microphone and camera
2. ✅ Close unnecessary apps
3. ✅ Do a practice run
4. ✅ Prepare backup examples
5. ✅ Have water nearby
6. ✅ Set phone to silent

---

## 🔧 Troubleshooting During Recording

### If Demo Fails:
- Stay calm
- Have screenshots/pre-recorded backup ready
- Explain what should happen
- Move on smoothly

### If You Make a Mistake:
- Pause briefly
- Start sentence again
- Edit can remove mistakes later

### If System Crashes:
- Have backup video of demo
- Can splice in during editing
- Explain live demo is challenging

---

## ✂️ Post-Recording Checklist

After recording:

✅ Review entire video  
✅ Check audio quality  
✅ Verify all demos are visible  
✅ Add title/credits if desired  
✅ Export in correct format  
✅ Test playback  
✅ Upload to submission platform  
✅ Confirm file uploaded successfully  

---

## 📤 Submission Format

**File Requirements:**
- Format: MP4 (H.264)
- Resolution: 1920x1080 or 1280x720
- Duration: 10-15 minutes
- File size: <500MB (compress if needed)
- Filename: ISY503_Assessment3_[TeamName].mp4

**Upload To:**
- Blackboard Assignment 3 submission link
- One team member uploads
- Include all team member names in comments

---

Good luck with your recording! 🎥🌟
