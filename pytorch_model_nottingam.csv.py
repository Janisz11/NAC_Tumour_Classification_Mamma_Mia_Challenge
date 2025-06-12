import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import csv

df2 = pd.read_excel('/home/kacjan7732/data_workspace/gtp_NCCN_based_filled_data_clinical_info.xlsx')
df1 = pd.read_csv('/home/kacjan7732/data_workspace/nottingham_summary.csv')
df_combined = df1.merge(df2[['patient_id', 'pcr']], on='patient_id', how='left')
df_combined = df_combined[df_combined['pcr'].notna()]


def preprocess(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.drop(columns=["pcr"])
    y = numeric_df["pcr"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32)
    )


class PCRNet(nn.Module):
    def __init__(self, input_dim):
        super(PCRNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(df: pd.DataFrame, csv_path="training_losses.csv"):
    X_train, y_train, X_val, y_val = preprocess(df)
    input_dim = X_train.shape[1]

    model = PCRNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    batch_size = 64

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size(0))
            train_loss = 0.0

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices].unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= (X_train.size(0) / batch_size)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
                preds = (val_outputs >= 0.5).float()
                correct = (preds.squeeze() == y_val).sum().item()
                accuracy = correct / y_val.size(0)

            writer.writerow([epoch + 1, train_loss, val_loss, accuracy])
            print(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model(df_combined)
