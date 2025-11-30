import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# LOAD TEST DATA
# ---------------------------
test_df = pd.read_csv("/kaggle/input/2025-sep-dl-gen-ai-project/test.csv")

# ---------------------------
# LOAD SAVED BERT MODEL
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert_model")
model = BertForSequenceClassification.from_pretrained("bert_model")
model.to(device)
model.eval()

# ---------------------------
# TOKENIZE TEST DATA
# ---------------------------
# Dynamic padding for efficiency
encodings = tokenizer(
    list(test_df["text"]),
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# Move input tensors to device
input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# ---------------------------
# BATCHED INFERENCE
# ---------------------------
batch_size = 32
predictions = []

with torch.no_grad():
    for i in range(0, len(test_df), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]

        logits = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        ).logits

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        predictions.extend(preds)

# ---------------------------
# CREATE SUBMISSION FILE
# ---------------------------
submission = pd.DataFrame(predictions, columns=EMOTIONS)
submission["id"] = test_df["id"]
submission = submission[["id"] + EMOTIONS]

# Save submission to Kaggle's working directory
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved successfully!")
