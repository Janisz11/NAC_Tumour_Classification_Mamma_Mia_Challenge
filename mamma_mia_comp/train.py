import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import csv
import os

from dataset import MammaMiaCompetitionDataset
from model import FusionPCRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ścieżki danych
SRC_ROOT = "/lustre/pd01/hpc-ljelen-1692966897/mamma_mia"
images_root = f"{SRC_ROOT}/images"
segmentation_root = f"{SRC_ROOT}/segmentations/expert"
clinical_xlsx = f"{SRC_ROOT}/clinical_and_imaging_info.xlsx"
splits_csv = f"{SRC_ROOT}/train_test_splits.csv"

splits_df = pd.read_csv(splits_csv).dropna(subset=["train_split", "test_split"])
clin_df = pd.read_excel(clinical_xlsx).dropna(subset=["pcr"])
valid_ids = set(clin_df["patient_id"].astype(str))

train_patients = [p for p in splits_df["train_split"].unique() if p in valid_ids]
val_patients   = [p for p in splits_df["test_split"].unique() if p in valid_ids]

train_dataset = MammaMiaCompetitionDataset(train_patients, images_root, clinical_xlsx, segmentation_root)
val_dataset   = MammaMiaCompetitionDataset(val_patients, images_root, clinical_xlsx, segmentation_root)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=2, num_workers=4)

model = FusionPCRNet(in_channels=6, tic_dim=4).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

log_path = "training_log.csv"
os.makedirs("models", exist_ok=True)
best_val_acc = 0.0

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_bal_acc", "val_bal_acc", "TP", "FP", "TN", "FN"])

epochs = 100
for epoch in range(epochs):
    model.train()
    total_train_loss, y_true_train, y_pred_train = 0, [], []

    for x, y, feats in train_loader:
        x, y, feats = x.to(device), y.to(device), feats.to(device)
        optimizer.zero_grad()
        logits = model(x, feats)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        y_true_train += y.cpu().tolist()
        y_pred_train += logits.argmax(dim=1).cpu().tolist()

    train_bal_acc = balanced_accuracy_score(y_true_train, y_pred_train)
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss, y_true_val, y_pred_val = 0, [], []
    with torch.no_grad():
        for x, y, feats in val_loader:
            x, y, feats = x.to(device), y.to(device), feats.to(device)
            logits = model(x, feats)
            loss = F.cross_entropy(logits, y)
            total_val_loss += loss.item()
            y_true_val += y.cpu().tolist()
            y_pred_val += logits.argmax(dim=1).cpu().tolist()

    val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
    avg_val_loss = total_val_loss / len(val_loader)

    # Confusion matrix
    y_true_tensor = torch.tensor(y_true_val)
    y_pred_tensor = torch.tensor(y_pred_val)
    TP = int(((y_true_tensor == 1) & (y_pred_tensor == 1)).sum())
    TN = int(((y_true_tensor == 0) & (y_pred_tensor == 0)).sum())
    FP = int(((y_true_tensor == 0) & (y_pred_tensor == 1)).sum())
    FN = int(((y_true_tensor == 1) & (y_pred_tensor == 0)).sum())

    print(f"Epoka {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Train BAcc: {train_bal_acc:.4f} | Val BAcc: {val_bal_acc:.4f} | TP: {TP} FP: {FP} TN: {TN} FN: {FN}")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, train_bal_acc, val_bal_acc, TP, FP, TN, FN])

    if val_bal_acc > best_val_acc:
        best_val_acc = val_bal_acc
        torch.save(model.state_dict(), "models/best_model.pth")
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
