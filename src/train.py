import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import RandLA_KPConv_UNet_Classifier


def compute_metrics(y_true, y_pred, num_classes):
    ious = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0
        ious.append(iou)

    mean_iou = np.mean(ious)
    macro_f1 = np.mean([
        (2 * np.sum((y_true == c) & (y_pred == c))) /
        (np.sum(y_true == c) + np.sum(y_pred == c) + 1e-6)
        for c in range(num_classes)
    ])

    return macro_f1, mean_iou, ious


def load_data(data_dir):
    train = torch.load(os.path.join(data_dir, "train.pt"))
    val = torch.load(os.path.join(data_dir, "val.pt"))

    X_train, y_train, _ = train
    X_val, y_val, _ = val

    return (
        TensorDataset(X_train, y_train),
        TensorDataset(X_val, y_val)
    )


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = load_data("data/processed")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = RandLA_KPConv_UNet_Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)

                y_true.append(yb.numpy())
                y_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        f1, miou, _ = compute_metrics(y_true, y_pred, 5)

        print(f"Epoch {epoch}: F1={f1:.3f}, mIoU={miou:.3f}")

    torch.save(model.state_dict(), "models/weights/ridar_net.pth")


if __name__ == "__main__":
    train()