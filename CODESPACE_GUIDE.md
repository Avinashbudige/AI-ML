# 🚀 Quick Start Guide for GitHub Codespace

## One-Click Setup

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=new-directory-update&repo=Avinashbudige/AI-ML)

## What Happens When You Open in Codespace?

1. **Automatic Environment Setup** (takes ~2-3 minutes)
   - Python 3.11 environment configured
   - All dependencies installed automatically
   - VS Code extensions loaded
   - Git configured

2. **Ready to Use**
   - Open integrated terminal (Ctrl+`)
   - Run the demo: `./run_demo.sh`

## Commands After Setup

### Quick Demo (Recommended First Run)
```bash
./run_demo.sh
```
Shows document loading, chunking, and configuration in action.

### Complete Pipeline Overview
```bash
python "Enterprise Agentic Analytics Assistant/document_loader/demo_complete.py"
```
Displays full pipeline with statistics and module breakdown.

### Run Test Suite
```bash
python "Enterprise Agentic Analytics Assistant/document_loader/test_offline.py"
```
Runs all offline tests to verify functionality.

### Manual Setup (if needed)
```bash
./.devcontainer/setup.sh
```

## What's Pre-Installed?

- ✅ Python 3.11
- ✅ PyPDF2 (PDF processing)
- ✅ sentence-transformers (BGE embeddings)
- ✅ faiss-cpu (vector similarity search)
- ✅ openai (OpenAI API, optional)
- ✅ All project dependencies from requirements.txt

## Codespace Features

- **Auto-save**: Files save automatically
- **Format on save**: Code auto-formatted with Black
- **Python IntelliSense**: Auto-completion and type hints
- **Jupyter Support**: Open and run .ipynb files directly
- **Git Integration**: Built-in source control panel

## File Explorer

Navigate to these key files:
```
📁 Enterprise Agentic Analytics Assistant/
  └─📁 document_loader/
     ├─ 📄 README.md           ← Full documentation
     ├─ 🎬 demo.py             ← Quick demo
     ├─ 🎬 demo_complete.py    ← Complete overview
     ├─ 🧪 test_offline.py     ← Test suite
     └─ 📚 (module files)
```

## Troubleshooting

### "Module not found" error?
```bash
pip install -r requirements.txt
```

### Need to reinstall dependencies?
```bash
pip install --force-reinstall PyPDF2 sentence-transformers faiss-cpu
```

### Terminal not opening?
Press `Ctrl+` ` (backtick) or go to Terminal → New Terminal

## Next Steps

1. ✅ Run `./run_demo.sh` to see it in action
2. ✅ Explore the README.md in document_loader/
3. ✅ Try modifying demo.py to test different features
4. ✅ Check out the example notebooks in notebooks/

---

**Need Help?** See [.devcontainer/README.md](.devcontainer/README.md) for detailed setup information.
