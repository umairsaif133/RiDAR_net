import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from model import RandLA_KPConv_UNet_Classifier


def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_test, y_test, ids = torch.load("data/processed/test.pt")

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = RandLA_KPConv_UNet_Classifier().to(device)
    model.load_state_dict(torch.load("models/weights/ridar_net.pth"))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    print("Inference done. Predictions shape:", all_preds.shape)


if __name__ == "__main__":
    run_inference()