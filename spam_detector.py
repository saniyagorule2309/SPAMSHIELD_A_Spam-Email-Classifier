"""
=============================================================
  SPAM EMAIL DETECTION SYSTEM
  Full ML Pipeline: Preprocessing → Training → Evaluation
=============================================================
Required Libraries:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    (nltk is optional – project works without it)
"""

import os, re, string, warnings, joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.svm                     import LinearSVC
from sklearn.metrics                 import (accuracy_score, confusion_matrix,
                                             classification_report,
                                             ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")

# ── Stopwords (no NLTK needed) ───────────────────────────────────────────────
_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}

def _simple_stem(word):
    if len(word) <= 3:
        return word
    for sfx in ("ing","tion","ness","ment","able","ful","less","ive","ous",
                "al","ly","ed","er","est","es","s"):
        if word.endswith(sfx) and len(word) - len(sfx) >= 3:
            return word[:-len(sfx)]
    return word

# ── Dataset ──────────────────────────────────────────────────────────────────
def load_dataset(path="data/spam.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
        print(f"[✓] Loaded '{path}'  →  {df.shape[0]} rows")
    else:
        print(f"[!] '{path}' not found – using demo dataset")
        df = _demo_dataset()
    return df

def _demo_dataset():
    spam = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "FREE entry in 2 a weekly competition to win FA Cup final tickets!",
        "URGENT: Your account has been compromised. Verify immediately.",
        "You have been selected for a cash prize of $5000. Reply WIN.",
        "Get rich quick! Earn $500 per day working from home. No experience needed.",
        "Claim your free iPhone now. Limited offer. Click the link below.",
        "Dear winner, you have been selected for a lottery prize of $10,000!",
        "Buy cheap Viagra online. No prescription needed. 50 percent discount today!",
        "Your loan has been approved! Get $50,000 instantly. No credit check.",
        "Hot singles in your area are waiting to meet you. Sign up free!",
        "You are a lucky winner of our weekly draw. Text PRIZE to 80888 now.",
        "Congratulations ur awarded $500 of CD gift vouchers or 2GB camera free.",
        "Our records indicate your computer is at risk. Call support now FREE.",
        "Double your income today! No investment needed. Guaranteed results!",
        "WINNER!! As a valued network customer you have been selected to receive a prize.",
    ] * 20
    ham = [
        "Hey, are you free for lunch tomorrow?",
        "Can you send me the report by end of day?",
        "Reminder: team meeting at 3 PM in conference room B.",
        "Happy birthday! Hope you have a wonderful day.",
        "I will be home late tonight. Don't wait up.",
        "Did you catch the game last night? Amazing ending!",
        "Please review the attached document and let me know your thoughts.",
        "Mom called. She wants you to call back when you get a chance.",
        "The package was delivered to your door this morning.",
        "Thanks for your help yesterday. Really appreciated it.",
        "Can we reschedule our 2pm meeting to 3pm?",
        "I have finished the assignment. Let me know what you think.",
        "We are having a birthday party for Jake this Saturday. You are invited!",
        "Don't forget to pick up milk on the way home.",
        "The project deadline has been moved to next Friday.",
    ] * 20
    labels = ["spam"]*len(spam) + ["ham"]*len(ham)
    texts  = spam + ham
    idx = np.random.default_rng(42).permutation(len(texts))
    return pd.DataFrame({"v1": np.array(labels)[idx], "v2": np.array(texts)[idx]})

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_dataframe(df):
    df = df[["v1","v2"]].copy()
    df.columns = ["label","text"]
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["label_num"] = df["label"].map({"spam":1,"ham":0})
    print(f"\n[✓] After preprocessing: {df.shape[0]} rows")
    print(df["label"].value_counts().to_string())
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    tokens = [_simple_stem(t) for t in text.split()
              if t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def apply_text_cleaning(df):
    df["clean_text"] = df["text"].apply(clean_text)
    print("[✓] Text cleaning complete")
    return df

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label_num"]
    print(f"[✓] TF-IDF matrix: {X.shape}")
    return X, y, vectorizer

# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(X, y, test_size=0.20, random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"[✓] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# ── Train ─────────────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Naive Bayes (MultinomialNB)":       MultinomialNB(),
        "Support Vector Machine (LinearSVC)": LinearSVC(C=1.0, max_iter=2000),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"[✓] Trained: {name}")
    return trained

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_models(trained_models, X_test, y_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    print("\n" + "═"*62 + "\n  MODEL EVALUATION RESULTS\n" + "═"*62)

    fig, axes = plt.subplots(1, len(trained_models),
                             figsize=(7*len(trained_models), 5))
    if len(trained_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Ham","Spam"])
        results[name] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
        print(f"\n▶  {name}\n   Accuracy: {acc*100:.2f}%")
        print(report)
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Ham","Spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy: {acc*100:.2f}%", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrices – Spam Classifier", fontsize=13, y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[✓] Confusion matrices → {cm_path}")

    # Accuracy bar chart
    names  = list(results.keys())
    scores = [v["accuracy"]*100 for v in results.values()]
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(names, scores, color=["#4C72B0","#DD8452"], edgecolor="white", width=0.45)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{score:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(max(0, min(scores)-5), 103)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[✓] Accuracy chart   → {acc_path}")
    return results

# ── Save ──────────────────────────────────────────────────────────────────────
def save_best_model(trained_models, results, vectorizer, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_name]
    model_path = os.path.join(output_dir, "spam_classifier.pkl")
    vec_path   = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer,  vec_path)
    print(f"\n[✓] Best model: {best_name}  ({results[best_name]['accuracy']*100:.2f}%)")
    print(f"[✓] Saved → {model_path} & {vec_path}")
    return best_name

# ── Predict ───────────────────────────────────────────────────────────────────
def predict_email(email_text,
                  model_path="models/spam_classifier.pkl",
                  vec_path="models/tfidf_vectorizer.pkl"):
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    features   = vectorizer.transform([clean_text(email_text)])
    pred       = model.predict(features)[0]
    return "SPAM" if pred == 1 else "HAM"

# ── CLI ───────────────────────────────────────────────────────────────────────
def run_cli():
    model_path = "models/spam_classifier.pkl"
    vec_path   = "models/tfidf_vectorizer.pkl"
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        print("[!] No trained model found – training now …\n")
        run_training_pipeline()
    print("\n" + "═"*55)
    print("  SPAM EMAIL CLASSIFIER  –  CLI Interface")
    print("  Type 'quit' to exit.")
    print("═"*55)
    while True:
        email = input("\nEnter email text: ").strip()
        if email.lower() in ("quit","exit","q"):
            print("Goodbye!")
            break
        if not email:
            continue
        result = predict_email(email, model_path, vec_path)
        icon   = "🚨" if result == "SPAM" else "✅"
        print(f"\n  Prediction: {icon} {result}\n")

# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_training_pipeline(dataset_path="data/spam.csv"):
    df = load_dataset(dataset_path)
    df = preprocess_dataframe(df)
    df = apply_text_cleaning(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_models = train_models(X_train, y_train)
    results = evaluate_models(trained_models, X_test, y_test)
    best = save_best_model(trained_models, results, vectorizer)
    print(f"\n{'═'*55}\n  ✅  Pipeline complete!  Best model: {best}\n{'═'*55}\n")
    return trained_models, results, vectorizer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_training_pipeline()
