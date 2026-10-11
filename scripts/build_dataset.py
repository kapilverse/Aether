from pathlib import Path
import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parent.parent

TOKENIZER = ROOT / "tokenizer" / "aether.model"

RAW = {
    "fineweb": ROOT / "data" / "raw" / "fineweb_edu.txt",
    "code": ROOT / "data" / "raw" / "code.txt",
    "wikipedia": ROOT / "data" / "raw" / "wikipedia.txt",
}

OUT = ROOT / "data" / "processed"

SEQ_LEN = 2048

# Exact target training mixture
TRAIN_TARGETS = {
    "fineweb": 424_904_598,
    "code": 84_980_920,
    "wikipedia": 56_653_946,
}

# Validation targets
VAL_TARGETS = {
    "fineweb": 4_291_966,
    "code": 1_439_914,
    "wikipedia": 675_464,
}


def tokenize_source(sp, path, target_tokens, output_path):
    """
    Tokenize a source and write exactly target_tokens (or slightly fewer
    if the source ends early) as uint16 token IDs.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0

    buffer = []

    print()
    print("=" * 60)
    print("Source:", path.name)
    print("Target tokens:", f"{target_tokens:,}")
    print("Output:", output_path)
    print("=" * 60)

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        with open(output_path, "wb") as out:

            for line_number, line in enumerate(f, 1):

                if not line:
                    continue

                ids = sp.encode(
                    line,
                    out_type=int,
                )

                remaining = target_tokens - written

                if len(ids) > remaining:
                    ids = ids[:remaining]

                if ids:
                    arr = np.asarray(
                        ids,
                        dtype=np.uint16,
                    )

                    arr.tofile(out)

                    written += len(ids)

                if line_number % 100000 == 0:
                    print(
                        f"Lines: {line_number:,} | "
                        f"Tokens: {written:,} / "
                        f"{target_tokens:,}"
                    )

                if written >= target_tokens:
                    break

    print(
        f"Finished: {written:,} tokens"
    )

    return written


def build_split(sp, split_name, targets):
    split_dir = OUT / split_name
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    for source, target in targets.items():

        output = (
            split_dir /
            f"{source}.bin"
        )

        written = tokenize_source(
            sp,
            RAW[source],
            target,
            output,
        )

        total += written

    print()
    print("=" * 60)
    print(split_name.upper(), "COMPLETE")
    print("=" * 60)
    print(
        "Total tokens:",
        f"{total:,}",
    )
    print(
        "Approx GB:",
        f"{total * 2 / 1024**3:.2f}",
    )


def main():

    print("Loading tokenizer...")

    sp = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER)
    )

    print(
        "Vocabulary:",
        sp.vocab_size(),
    )

    build_split(
        sp,
        "train",
        TRAIN_TARGETS,
    )

    build_split(
        sp,
        "val",
        VAL_TARGETS,
    )

    print()
    print("=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)

    print()
    print("Created:")
    print("  data/processed/train/")
    print("  data/processed/val/")


if __name__ == "__main__":
    main()