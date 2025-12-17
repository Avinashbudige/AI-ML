# GitHub Codespace Setup for EA³ Document Loader

This directory contains the development container configuration for running the Enterprise Agentic Analytics Assistant (EA³) Document Loader module in GitHub Codespaces.

## 🚀 Quick Start in Codespace

### Option 1: Automatic Setup
When you open this repository in a Codespace, it will automatically:
1. Set up a Python 3.11 environment
2. Install all required dependencies from `requirements.txt`
3. Display a welcome message with instructions

### Option 2: Manual Setup
If you need to set up manually, run:

```bash
./.devcontainer/setup.sh
```

This will:
- Install dependencies (PyPDF2, sentence-transformers, faiss-cpu, openai)
- Run the offline test suite
- Execute the demonstration script

## 📋 What's Included

### Python Extensions
- Python language support with IntelliSense
- Jupyter notebooks support
- Auto-formatting with Black
- Linting with Pylint

### Configuration
- Python 3.11 base image
- Git pre-configured
- Zsh shell with Oh My Zsh
- Auto-save enabled
- Format on save enabled

## 🎯 Running the Document Loader

### Quick Demo
```bash
python "Enterprise Agentic Analytics Assistant/document_loader/demo.py"
```

### Complete Pipeline Overview
```bash
python "Enterprise Agentic Analytics Assistant/document_loader/demo_complete.py"
```

### Run Test Suite
```bash
python "Enterprise Agentic Analytics Assistant/document_loader/test_offline.py"
```

### View Documentation
```bash
cat "Enterprise Agentic Analytics Assistant/document_loader/README.md"
```

## 📦 Dependencies

The following packages are automatically installed:
- `PyPDF2` - PDF processing
- `sentence-transformers` - BGE embeddings (local)
- `faiss-cpu` - Vector similarity search
- `openai` - OpenAI API integration (optional)
- `numpy`, `pandas`, `scikit-learn` - Data processing

## 🔧 Customization

### Modify Python Version
Edit `devcontainer.json` and change:
```json
"image": "mcr.microsoft.com/devcontainers/python:3.11"
```

### Add More Extensions
Add to the `extensions` array in `devcontainer.json`:
```json
"extensions": [
  "ms-python.python",
  // Add your extensions here
]
```

### Change Post-Create Commands
Modify `postCreateCommand` in `devcontainer.json`:
```json
"postCreateCommand": "pip install -r requirements.txt && echo 'Custom setup complete'"
```

## 🐛 Troubleshooting

### Dependencies Not Installing
Run manually:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Module Import Errors
Ensure you're in the repository root:
```bash
cd /workspaces/AI-ML
python "Enterprise Agentic Analytics Assistant/document_loader/demo.py"
```

### Need to Reinstall Dependencies
```bash
pip install --force-reinstall PyPDF2 sentence-transformers faiss-cpu
```

## 📚 Additional Resources

- [Main README](../Enterprise%20Agentic%20Analytics%20Assistant/document_loader/README.md)
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)
- [Project README](../README.md)

## 💡 Tips

1. **Use the integrated terminal**: Press `` Ctrl+` `` to open
2. **Run Jupyter notebooks**: Files ending in `.ipynb` can be opened directly
3. **Format code**: Right-click → Format Document (or save to auto-format)
4. **Git integration**: Use the Source Control panel on the left

---

Happy coding in your Codespace! 🚀
