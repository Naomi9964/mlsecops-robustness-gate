# Deterministic synthetic dataset: class 0 = disc blob, class 1 = square blob.
# Generated on the fly so the demo runs offline with no downloads.

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_dataset(n=2000, img=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.zeros(n, 1, img, img)
    ys = torch.randint(0, 2, (n,), generator=g)
    yy, xx = torch.meshgrid(torch.arange(img), torch.arange(img), indexing="ij")
    for i in range(n):
        cx = int(torch.randint(4, img - 4, (1,), generator=g))
        cy = int(torch.randint(4, img - 4, (1,), generator=g))
        if ys[i] == 0:
            shape = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 9).float()  # disc, r=3
        else:
            shape = torch.zeros(img, img)
            shape[cy - 3:cy + 3, cx - 3:cx + 3] = 1.0  # 6x6 square
        noise = torch.rand(img, img, generator=g) * 0.2
        xs[i, 0] = (shape + noise).clamp(0.0, 1.0)
    return TensorDataset(xs, ys.long())


def make_loaders(n=2000, batch=128, seed=0):
    ds = make_dataset(n=n, seed=seed)
    train_ds, test_ds = random_split(
        ds, [int(0.8 * n), n - int(0.8 * n)],
        generator=torch.Generator().manual_seed(seed),
    )
    mk = lambda d: DataLoader(d, batch_size=batch, shuffle=True)
    return mk(train_ds), mk(test_ds)
