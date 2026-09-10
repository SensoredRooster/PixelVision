from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.core.dataset_loader import AntiCheatMLDatasetLoader


class PixelVisionClassifier(nn.Module):
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.network(x)


class AntiCheatModelTrainer:
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODEL] Training pipeline running on target device: {self.device}")

    def train_and_export(self, X_data: np.ndarray, y_data: np.ndarray, epochs: int = 20, batch_size: int = 4):
        if len(X_data) == 0:
            print("[ERROR] Training cancelled: the dataset matrices are empty.")
            return

        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = PixelVisionClassifier(input_dim=X_data.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        print(f"[TRAIN] Beginning training execution loop across {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f" -> Epoch {epoch + 1}/{epochs} | Aggregated Loss: {epoch_loss / len(dataloader):.4f}")

        model.eval()
        onnx_path = os.path.join(self.model_dir, "pixelvision_detector.onnx")
        dummy_input = torch.randn(1, X_data.shape[1]).to(self.device)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_features"],
            output_names=["cheat_probabilities"],
        )
        print(f"[EXPORT] Success! Production-ready model written to: {onnx_path}")


if __name__ == "__main__":
    print("[SYSTEM] Pulling dataset states from compiler loader pipeline...")
    dataset_loader = AntiCheatMLDatasetLoader()
    X, y = dataset_loader.compile_training_tensors()

    if len(X) == 0:
        print("[WARNING] No local clips found. Simulating synthetic clip-level features for a pipeline check...")
        X = np.array(
            [
                [4.0, 0.72, 3.5, 0.0, 0.4, 6.0],
                [5.0, 0.68, 4.1, 0.0, 0.5, 7.0],
                [22.0, 0.99, 0.1, 4.0, 0.8, 31.0],
                [19.0, 0.98, 0.2, 3.0, 0.7, 29.0],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1], dtype=np.int64)

    trainer = AntiCheatModelTrainer()
    trainer.train_and_export(X, y, epochs=15)
