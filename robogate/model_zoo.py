# Sample model + training helper. Bring your own model by saving a
# torch.nn.Module state_dict and loading it in your own script.

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    # fully-connected baseline: no convolution, no spatial inductive bias.
    # Tests whether locality/weight-sharing helps robustness.
    def __init__(self, img=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img * img, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class DeepCNN(nn.Module):
    # deeper than TinyCNN: three conv blocks instead of two.
    # Tests whether depth helps robustness.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# model zoo: name -> model class, used by examples/compare_models.py
MODELS = {
    "tinycnn": TinyCNN,
    "mlp": MLP,
    "deepcnn": DeepCNN,
}


def train(model, loader, epochs=5, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model
