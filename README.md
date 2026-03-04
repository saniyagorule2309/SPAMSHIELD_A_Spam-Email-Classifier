# 📧 Spam Email Detection System

A complete Machine Learning project that classifies emails as **Spam** or **Ham (Not Spam)** using NLP and two ML models: Naive Bayes and SVM.

---

## 📁 Project Structure

```
spam_detector/
├── spam_detector.py      # Full ML pipeline (preprocessing → training → evaluation)
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
├── data/
│   └── spam.csv          # Dataset (download separately – see below)
├── models/               # Auto-created after training
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
└── outputs/              # Auto-created after training
    ├── confusion_matrices.png
    └── accuracy_comparison.png
```

---

## ⚙️ Installation

```bash
# 1. Clone / download the project folder
cd spam_detector

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

Download the **SMS Spam Collection** dataset:

- **Kaggle:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Save the file as `data/spam.csv`

> **No dataset?** No problem — the project automatically generates a small demo dataset so you can run everything immediately.

---

## 🚀 How to Run

### Option A – Training Pipeline Only

```bash
python spam_detector.py
```

This will:
1. Load & preprocess the dataset
2. Clean text (lowercase, remove stopwords, stem)
3. Extract TF-IDF features
4. Train Naive Bayes + SVM models
5. Evaluate & compare both models
6. Save the best model to `models/`
7. Save evaluation charts to `outputs/`

---

### Option B – Command-Line Interface (CLI)

```bash
python spam_detector.py --cli
```

Type any email text and get an instant prediction:

```
Enter email text: Congratulations! You've won a $1000 prize!

  Prediction: 🚨 SPAM
```

---

### Option C – Streamlit Web App

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501` where you can:
- Paste any email and classify it
- Try built-in sample emails
- View model evaluation charts

---

## 🔬 ML Pipeline Explained

| Step | Description |
|------|-------------|
| **1. Load Data** | Reads `spam.csv` (latin-1 encoding), or generates demo data |
| **2. Preprocess** | Keep `label` + `text` columns, drop duplicates, encode labels (spam=1, ham=0) |
| **3. Clean Text** | Lowercase → remove URLs/numbers/punctuation → tokenise → remove stopwords → stem |
| **4. TF-IDF** | `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` |
| **5. Split** | 80% train / 20% test, stratified |
| **6. Train** | MultinomialNB + LinearSVC |
| **7. Evaluate** | Accuracy, confusion matrix, precision/recall/F1 |
| **8. Save** | Best model + vectorizer saved via `joblib` |

---

## 📊 Expected Results (on full SMS Spam dataset)

| Model | Accuracy |
|-------|----------|
| Naive Bayes (MultinomialNB) | ~97–98% |
| SVM (LinearSVC) | ~98–99% |

---

## 🛠️ Key Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `nltk` | Stopwords, stemming |
| `scikit-learn` | TF-IDF, models, metrics |
| `matplotlib / seaborn` | Evaluation charts |
| `joblib` | Save/load model |
| `streamlit` | Web UI |

---

## ❓ Troubleshooting

| Issue | Fix |
|-------|-----|
| `UnicodeDecodeError` | Already handled — CSV loaded with `encoding="latin-1"` |
| NLTK data missing | Script auto-downloads `stopwords`, `punkt` |
| Model not found (CLI) | CLI auto-trains the model first |
| `streamlit` not found | Run `pip install streamlit` |
