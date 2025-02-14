#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
python -m pip install --upgrade pip
#!/usr/bin/env bash
set -o errexit  # Exit on error

# Use Python 3.10
pyenv global 3.10 || echo "Python 3.10 not found"

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Install dependencies
pip install -r requirements.txt

# Install additional required packages
pip install wheel setuptools

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run setup and training
python setup.py
python train.py