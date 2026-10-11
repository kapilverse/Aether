from pathlib import Path
import torch

from training.data import create_train_loader


def main():

    root = Path(__file__).resolve().parent.parent

    loader = create_train_loader(
        data_dir=root / "data" / "processed" / "train",
        batch_size=2,
        seq_len=2048,
        num_workers=0,
    )

    x, y = next(iter(loader))

    print("X shape:", tuple(x.shape))
    print("Y shape:", tuple(y.shape))

    print("X dtype:", x.dtype)
    print("Y dtype:", y.dtype)

    print("X min:", x.min().item())
    print("X max:", x.max().item())

    assert x.shape == (2, 2048)
    assert y.shape == (2, 2048)

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    assert x.min() >= 0
    assert x.max() < 32000
    assert y.min() >= 0
    assert y.max() < 32000

    print()
    print("DataLoader: PASS")
    print("Shape: PASS")
    print("Token range: PASS")


if __name__ == "__main__":
    main()