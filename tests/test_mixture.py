from pathlib import Path

import numpy as np
import torch

from training.data import TokenStreamDataset


def main():

    root = Path(__file__).resolve().parent.parent

    files = [
        root / "data" / "processed" / "train" / "fineweb.bin",
        root / "data" / "processed" / "train" / "code.bin",
        root / "data" / "processed" / "train" / "wikipedia.bin",
    ]

    dataset = TokenStreamDataset(
        files=files,
        weights=[0.75, 0.15, 0.10],
        seq_len=2048,
        seed=42,
    )

    rng = np.random.default_rng(42)

    counts = np.zeros(3, dtype=np.int64)

    samples = 10000

    for _ in range(samples):

        index = rng.choice(
            3,
            p=[0.75, 0.15, 0.10],
        )

        counts[index] += 1

    percentages = (
        counts / samples * 100
    )

    print("Mixture test")
    print("=" * 40)

    print(
        f"FineWeb-Edu : {percentages[0]:.2f}%"
    )
    print(
        f"Code        : {percentages[1]:.2f}%"
    )
    print(
        f"Wikipedia   : {percentages[2]:.2f}%"
    )

    assert 73.0 < percentages[0] < 77.0
    assert 13.0 < percentages[1] < 17.0
    assert 8.0 < percentages[2] < 12.0

    print()
    print("Mixture sampling: PASS")


if __name__ == "__main__":
    main()