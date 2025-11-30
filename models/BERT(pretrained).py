# ============================
# BERT Multi-Label Emotion Classification
# ============================
import os
os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATE"] = "1"
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import f1_score

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# PREPARE TOKENIZER
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding=False   # dynamic padding handled by DataCollator
    )

# ---------------------------
# METRICS FUNCTION
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))      # sigmoid for multi-label
    preds = (probs > 0.5).astype(int)      # threshold 0.5
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}

# ---------------------------
# PREPARE DATASET
# ---------------------------
df2 = train_df.copy()
# convert labels → float for multi-label classification
df2["labels"] = df2[EMOTIONS].astype(float).values.tolist()

dataset = Dataset.from_pandas(df2)
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(tokenize, batched=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)
model.to(device)

# ---------------------------
# TRAINING ARGUMENTS
# ---------------------------
args = TrainingArguments(
    output_dir="./bert",
    eval_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    fp16=True,          # mixed precision if GPU available
    report_to="none",
    run_name="bert-pretrained"
)

# ---------------------------
# DATA COLLATOR
# ---------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# TRAINER
# ---------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ---------------------------
# TRAIN MODEL
# ---------------------------
trainer.train()

# ---------------------------
# SAVE MODEL + TOKENIZER
# ---------------------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
print("\n🎉 Model and tokenizer saved successfully!")
