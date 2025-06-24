# Fake-News-Detector
A terminal-based tool that classifies news headlines or articles as **REAL** or **FAKE** using TF-IDF and a tuned Naive Bayes model, achieving ~96% cross-validation accuracy.
## 🔍 Overview
In the era of information overload and misinformation, this project automatically flags whether a piece of news is real or fake.  
It uses:
- **Text cleaning & lemmatization** (NLTK)  
- **TF-IDF** feature extraction (unigrams + bigrams)  
- **Multinomial Naive Bayes** tuned via `GridSearchCV`  

---

## ✨ Features
- ✅ Automatic data loading & preprocessing  
- ✅ Hyperparameter tuning with cross-validation  
- ✅ Command-line interface for real-time predictions  
- ✅ (Optional) Learning-curve & heatmap visualizations  


---

## 🎬 Demo

```bash
$ python predict.py
📢 Fake News Detector
> “New study shows chocolate improves brain health.”
🧠 Prediction: REAL

> “Aliens found living in Antarctica, scientists confirm.”
🧠 Prediction: FAKE
