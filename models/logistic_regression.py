from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd

# ------------------------------
# 1. Convert text → TF-IDF features
# ------------------------------
tfidf = TfidfVectorizer(
    max_features=12000,     # more features = better performance
    ngram_range=(1,2),      # include bigrams for better emotion capture
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(train_df["clean_text"])

# ------------------------------
# 2. Train one classifier per emotion (multi-label)
# ------------------------------
models = {}

for emo in EMOTIONS:
    clf = LogisticRegression(
        max_iter=2000,         # increase iterations → more stable training
        class_weight="balanced", # handles imbalance in emotions
        solver="lbfgs"
    )
    
    clf.fit(X_tfidf, train_df[emo])
    models[emo] = clf

# ------------------------------
# 3. Predict on validation set
# ------------------------------
X_val_tfidf = tfidf.transform(X_val)

preds = {}
for emo in EMOTIONS:
    preds[emo] = models[emo].predict(X_val_tfidf)

preds_df = pd.DataFrame(preds)

# ------------------------------
# 4. Macro F1 Score
# ------------------------------
f1 = f1_score(y_val, preds_df, average="macro")
print("TF-IDF + Logistic Regression Macro F1:", f1)