import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_mamma_mia import MammaMiaDataset
from datetime import datetime
import pandas as pd
import csv
from sklearn.metrics import classification_report, accuracy_score
from torch.nn.functional import pad
import webbrowser

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels):
        super(Simple3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(256, 1024)
        self.fc_out = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

def pad_collate(batch):
    images, labels = zip(*batch)
    desired_c = 11
    max_shape = [max(s) for s in zip(*[img.shape[1:] for img in images])]
    padded_images = []
    for img in images:
        c, d, h, w = img.shape
        pad_c = desired_c - c
        pad_d = max_shape[0] - d
        pad_h = max_shape[1] - h
        pad_w = max_shape[2] - w
        padded = pad(img, (0, pad_w, 0, pad_h, 0, pad_d, 0, max(0, pad_c)))
        if pad_c < 0:
            padded = padded[:desired_c]  # cut off excess channels
        padded_images.append(padded)
    return torch.stack(padded_images), torch.tensor(labels).float()

def train_model():
    SRC_ROOT = "/lustre/pd01/hpc-ljelen-1692966897/mamma_mia"
    IMG_DIR = os.path.join(SRC_ROOT, "images")
    SEG_EXPERT_DIR = os.path.join(SRC_ROOT, "segmentations", "expert")
    CLINICAL_XLSX = os.path.join(SRC_ROOT, "clinical_and_imaging_info.xlsx")

    LOG_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_METRICS = os.path.join(LOG_DIR, "metrics.csv")

    dataset = MammaMiaDataset(
        image_root=IMG_DIR,
        seg_root=SEG_EXPERT_DIR,
        clinical_xlsx=CLINICAL_XLSX,
        margin=10,
        crop=True,
        use_diffs=True,
        skip_missing_labels=True
    )

    val_split = 0.2
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, collate_fn=pad_collate)

    in_channels = 11

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN(in_channels).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")
    best_val_loss = float("inf")

    if not os.path.exists(CSV_METRICS):
        with open(CSV_METRICS, "w") as tmp:
            pass
    try:
        webbrowser.open(f"file://{CSV_METRICS}")
    except:
        pass

    with open(log_path, "w") as f:
        for epoch in range(1, 200):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).unsqueeze(1)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * x.size(0)
                    preds = (outputs > 0.5).int().cpu().numpy()
                    y_pred.extend(preds)
                    y_true.extend(y.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            log_line = f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Acc = {accuracy:.4f}\n"
            print(log_line, end="")
            f.write(log_line)

            fieldnames = ["accuracy"] + [
                f"{label}_{metric}"
                for label in report
                if isinstance(report[label], dict)
                for metric in report[label]
            ]

            row = {"accuracy": accuracy}
            for label in report:
                if isinstance(report[label], dict):
                    for metric in report[label]:
                        row[f"{label}_{metric}"] = report[label][metric]

            file_exists = os.path.isfile(CSV_METRICS)
            with open(CSV_METRICS, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pt"))
                f.write("Model saved.\n")

if __name__ == "__main__":
    train_model()
