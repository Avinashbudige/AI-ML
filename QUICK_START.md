# Quick Start Guide

This guide will help you get started with the AI/ML/DL Interview Preparation Curriculum.

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for deep learning notebooks)
- Basic understanding of Python programming
- Familiarity with command line/terminal

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Avinashbudige/AI-ML.git
cd AI-ML
```

### 2. Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

**Option A: Install all dependencies (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install minimal dependencies first**
```bash
# Core libraries only
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Add deep learning later
pip install tensorflow torch torchvision
```

### 4. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Your browser will open automatically. Navigate to the notebooks and start learning!

## Learning Path

### Complete Beginner Path (Start here if new to ML)
1. Start with `01_numpy_pandas_basics.ipynb`
2. Continue sequentially through `02` to `07`
3. This covers all foundational machine learning concepts

### Intermediate Path (Have ML basics)
1. Start with `04_ensemble_methods.ipynb`
2. Continue through `05`, `06`, `07`
3. Then move to deep learning notebooks `08`-`12`

### Interview Prep Sprint (2-3 weeks)
- Week 1: Notebooks 01-04 (Foundations + Classical ML)
- Week 2: Notebooks 05-07 (Advanced ML + Unsupervised)
- Week 3: Notebooks 08-12 (Deep Learning specializations)

### Specialization Paths

**Machine Learning Engineer:**
- Focus on: 01, 02, 03, 04, 05, 06, 07

**Deep Learning Engineer:**
- Prerequisites: 01, 02, 03
- Focus on: 08, 09, 10, 11

**Computer Vision Engineer:**
- Prerequisites: 01, 02, 03, 08
- Focus on: 09

**NLP Engineer:**
- Prerequisites: 01, 02, 03, 08
- Focus on: 10

**Research Scientist:**
- All notebooks 01-12

## Tips for Success

### 1. Active Learning
- Don't just read - type out the code yourself
- Modify examples and see what happens
- Implement algorithms from scratch for deeper understanding

### 2. Practice Exercises
- Complete the practice exercises in each notebook
- Work on Kaggle datasets to apply concepts
- Build small projects for your portfolio

### 3. Interview Preparation
- Review interview questions in each notebook
- Practice explaining concepts out loud
- Understand the "why" and "when" not just "how"
- Be ready to discuss trade-offs between approaches

### 4. Time Management
- Spend 2-3 hours per notebook minimum
- Take breaks between topics
- Review previous topics regularly
- Keep notes on key concepts

## Common Issues and Solutions

### Issue: Installation fails
**Solution:** 
- Update pip: `pip install --upgrade pip`
- Install packages one at a time
- Check Python version compatibility

### Issue: Out of memory errors
**Solution:**
- Close other applications
- Use smaller datasets for practice
- Reduce batch sizes in deep learning notebooks

### Issue: TensorFlow/PyTorch errors
**Solution:**
- Check GPU drivers if using GPU
- Install CPU versions only if no GPU
- Use Google Colab as alternative

### Issue: Slow notebook execution
**Solution:**
- Use smaller sample sizes for initial learning
- Skip time-intensive visualization for now
- Consider using cloud platforms (Colab, Kaggle)

## Additional Resources

### Online Platforms
- **Kaggle**: Practice with real datasets and competitions
- **Google Colab**: Free GPU access for deep learning
- **HackerRank/LeetCode**: ML-specific coding challenges

### Documentation
- **Scikit-learn**: https://scikit-learn.org/
- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **Pandas**: https://pandas.pydata.org/

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Video Courses
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS229, CS230, CS231n

## Getting Help

1. **GitHub Issues**: Report bugs or ask questions
2. **Stack Overflow**: Search for specific error messages
3. **Documentation**: Always check official docs first
4. **Community Forums**: Reddit r/MachineLearning, r/learnmachinelearning

## Progress Tracking

Create a checklist to track your progress:

```
Foundations:
[ ] 01 - NumPy and Pandas Basics
[ ] 02 - Statistics and EDA
[ ] 03 - Supervised Learning

Intermediate:
[ ] 04 - Ensemble Methods
[ ] 05 - Feature Engineering
[ ] 06 - Hyperparameter Tuning
[ ] 07 - Unsupervised Learning

Advanced:
[ ] 08 - Neural Networks
[ ] 09 - Computer Vision
[ ] 10 - NLP
[ ] 11 - GANs
[ ] 12 - Reinforcement Learning
```

## Next Steps

After completing this curriculum:

1. **Build Projects**: Apply knowledge to real-world problems
2. **Contribute to Open Source**: Gain practical experience
3. **Read Research Papers**: Stay updated with latest developments
4. **Participate in Competitions**: Kaggle, DrivenData, AIcrowd
5. **Network**: Join ML communities and attend meetups
6. **Interview Practice**: Mock interviews and coding challenges

## Contact

For questions or suggestions:
- GitHub: [@Avinashbudige](https://github.com/Avinashbudige)
- Open an issue in the repository

---

**Good luck with your learning journey! 🚀**

Remember: Consistency is key. Even 30 minutes of focused practice daily is better than cramming.
