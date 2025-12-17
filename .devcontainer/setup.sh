#!/bin/bash

# Codespace Quick Start Script for Document Loader Module
# This script sets up and runs the Enterprise Agentic Analytics Assistant document loader

set -e

echo "=================================================="
echo "EA³ Document Loader - Codespace Quick Start"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "Enterprise Agentic Analytics Assistant/document_loader" ]; then
    echo "❌ Error: document_loader directory not found"
    echo "Please run this script from the repository root"
    exit 1
fi

echo "📦 Step 1: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet PyPDF2 sentence-transformers faiss-cpu openai

echo "✅ Dependencies installed"
echo ""

echo "🧪 Step 2: Running offline tests..."
python "Enterprise Agentic Analytics Assistant/document_loader/test_offline.py"

echo ""
echo "🚀 Step 3: Running demonstration..."
python "Enterprise Agentic Analytics Assistant/document_loader/demo.py"

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Available commands:"
echo "  • Quick demo:      python 'Enterprise Agentic Analytics Assistant/document_loader/demo.py'"
echo "  • Full overview:   python 'Enterprise Agentic Analytics Assistant/document_loader/demo_complete.py'"
echo "  • Run tests:       python 'Enterprise Agentic Analytics Assistant/document_loader/test_offline.py'"
echo "  • View docs:       cat 'Enterprise Agentic Analytics Assistant/document_loader/README.md'"
echo ""
echo "Happy coding! 🎉"
