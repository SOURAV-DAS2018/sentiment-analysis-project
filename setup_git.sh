#!/bin/bash

# Git Setup Script for Sentiment Analysis Project
# This script initializes a Git repository and prepares it for GitHub

echo "=========================================="
echo "Git Repository Setup"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed"
    echo "Please install Git first: https://git-scm.com/downloads"
    exit 1
fi

echo "✓ Git is installed"
echo ""

# Initialize git repository
echo "Initializing Git repository..."
git init

echo "✓ Git repository initialized"
echo ""

# Create .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Models and Data (too large for GitHub)
models/*.h5
models/*.pkl
data/positive/
data/negative/
*.csv
*.tsv

# MacOS
.DS_Store

# Training outputs
training_history.png
*.log

# Virtual Environment
venv/
env/
EOF

echo "✓ .gitignore created"
echo ""

# Add all files
echo "Adding files to Git..."
git add .

echo "✓ Files added"
echo ""

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Sentiment Analysis Project

- Complete data preprocessing pipeline
- Neural network architecture (LSTM-based)
- Training and evaluation scripts
- Web interface with Flask
- Comprehensive documentation
- Ethical considerations document"

echo "✓ Initial commit created"
echo ""

echo "=========================================="
echo "✓ Git Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create a repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it: sentiment-analysis-project"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Link your local repository to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-project.git"
echo ""
echo "3. Push your code:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Your GitHub repository URL will be:"
echo "   https://github.com/YOUR_USERNAME/sentiment-analysis-project"
echo ""
echo "=========================================="
