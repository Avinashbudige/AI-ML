#!/bin/bash

# Quick launcher for Document Loader Demo in Codespace
# Usage: ./run_demo.sh

echo "🚀 Launching EA³ Document Loader Demo..."
echo ""

# Navigate to repo root if needed
cd "$(dirname "$0")"

# Check if dependencies are installed
if ! python -c "import PyPDF2" 2>/dev/null; then
    echo "📦 Installing dependencies first..."
    pip install --quiet PyPDF2 sentence-transformers faiss-cpu openai
    echo "✅ Dependencies installed"
    echo ""
fi

# Run the demo
python "Enterprise Agentic Analytics Assistant/document_loader/demo.py"

echo ""
echo "💡 Next steps:"
echo "   - Try: python 'Enterprise Agentic Analytics Assistant/document_loader/demo_complete.py'"
echo "   - Test: python 'Enterprise Agentic Analytics Assistant/document_loader/test_offline.py'"
echo "   - Docs: cat 'Enterprise Agentic Analytics Assistant/document_loader/README.md'"
