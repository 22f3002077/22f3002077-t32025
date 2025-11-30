import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------
# DEVICE SELECTION
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class LSTMEmotion(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_labels=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMEmotion(len(vocab)).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# DATASETS + LOADERS
# ---------------------------
train_ds = TextDataset(X_train, y_train, tokenizer_simple)
val_ds   = TextDataset(X_val, y_val, tokenizer_simple)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=64, num_workers=0)

# ---------------------------
# DEBUG: TEST THE DATALOADER
# ---------------------------
print("\nTesting if DataLoader works...")

for i, batch in enumerate(train_loader):
    print("✔ First batch loaded successfully!")
    break

# ---------------------------
# EVALUATION FUNCTION
# ---------------------------
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).int()

            preds.append(pred.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    return f1_score(trues, preds, average="macro")

# ---------------------------
# TRAINING LOOP
# ---------------------------
log_every = 40

for epoch in range(15):  # keep 3 epochs for test run
    model.train()
    print(f"\n🔵 Epoch {epoch+1} starting...")

    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            pct = (step+1) / len(train_loader) * 100
            print(f"  ➤ Step {step+1}/{len(train_loader)} ({pct:.1f}%) | Loss={loss.item():.4f}")

    f1 = evaluate(model, val_loader)
    print(f"✅ Epoch {epoch+1} — Validation Macro F1: {f1:.4f}")

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), "lstm_model_fast.pth")
print("\n🎉 Training done! Model saved as lstm_model_fast.pth")
