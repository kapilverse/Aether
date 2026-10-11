from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class TokenStreamDataset(IterableDataset):

    def __init__(
        self,
        files,
        weights,
        seq_len=2048,
        seed=42,
    ):
        super().__init__()

        self.files = [
            Path(f) for f in files
        ]

        self.weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        self.weights /= self.weights.sum()

        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id

        rng = np.random.default_rng(
            self.seed + worker_id
        )

        while True:

            # Weighted source selection
            file_index = rng.choice(
                len(self.files),
                p=self.weights,
            )

            path = self.files[file_index]

            tokens = np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
            )

            max_start = (
                len(tokens)
                - self.seq_len
                - 1
            )

            start = rng.integers(
                0,
                max_start,
            )

            x = np.asarray(
                tokens[
                    start:
                    start + self.seq_len
                ],
                dtype=np.int64,
            )

            y = np.asarray(
                tokens[
                    start + 1:
                    start + self.seq_len + 1
                ],
                dtype=np.int64,
            )

            yield (
                torch.from_numpy(x),
                torch.from_numpy(y),
            )


def create_train_loader(
    data_dir,
    batch_size,
    seq_len=2048,
    num_workers=0,
):

    data_dir = Path(data_dir)

    files = [
        data_dir / "fineweb.bin",
        data_dir / "code.bin",
        data_dir / "wikipedia.bin",
    ]

    for path in files:
        if not path.exists():
            raise RuntimeError(
                f"Missing dataset file: {path}"
            )

    # EXACT 75 / 15 / 10 sampling
    weights = [
        0.75,
        0.15,
        0.10,
    ]

    dataset = TokenStreamDataset(
        files=files,
        weights=weights,
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )