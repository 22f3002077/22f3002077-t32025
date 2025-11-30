import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

# Load training data
train_df = pd.read_csv("/kaggle/input/2025-sep-dl-gen-ai-project/train.csv")

# Emotion labels (multi-label)
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]

# Simple text cleaning function
def clean_text(text):
    text = text.lower()                                           # convert to lowercase
    text = re.sub(r"http\S+", "", text)                          # remove URLs
    text = re.sub(r"[^a-zA-Z ]", " ", text)                      # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()                     # remove extra spaces
    return text

# Apply cleaning
train_df["clean_text"] = train_df["text"].apply(clean_text)

# Train-validation split (80% training / 20% validation)
X_train, X_val, y_train, y_val = train_df["clean_text"], train_df[EMOTIONS], train_df[EMOTIONS], train_df[EMOTIONS]

X_train, X_val, y_train, y_val = train_test_split(
    train_df["clean_text"],
    train_df[EMOTIONS],
    test_size=0.2,
    random_state=42
)
