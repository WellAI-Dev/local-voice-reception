#!/bin/bash
# Local development setup script for Apple Silicon (M4 Max)
# This script sets up the local development environment with MPS support

set -e

echo "========================================"
echo "Local Voice Reception AI - Setup Script"
echo "========================================"

# Find Python 3.11 (preferred for Vosk compatibility)
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MINOR=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f2)

echo "Found Python: $PYTHON_VERSION"

# Warn if using Python 3.12+
if [[ "$PYTHON_MINOR" -ge 12 ]]; then
    echo "⚠️  Warning: Python 3.12+ detected. Vosk may have compatibility issues."
    echo "   Recommended: Install Python 3.11 via pyenv or brew"
    echo "   brew install python@3.11"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with MPS support (Apple Silicon)
echo "Installing PyTorch with MPS support..."
pip install torch torchaudio

# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Install remaining dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama
fi

echo "✓ Ollama installed"

# Start Ollama service and pull model
echo "Starting Ollama and pulling Qwen2.5:7b model..."
ollama serve &
sleep 3
ollama pull qwen2.5:7b

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download Vosk model:"
echo "   python scripts/download_models.py"
echo ""
echo "2. Activate environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Run the application:"
echo "   python src/app.py"
echo ""
