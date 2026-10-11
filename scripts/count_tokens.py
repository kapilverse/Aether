from pathlib import Path
import sentencepiece as spm


ROOT = Path(__file__).resolve().parent.parent

TOKENIZER = ROOT / "tokenizer" / "aether.model"

FILES = {
    "FineWeb-Edu": ROOT / "data" / "raw" / "fineweb_edu.txt",
    "Code": ROOT / "data" / "raw" / "code.txt",
    "Wikipedia": ROOT / "data" / "raw" / "wikipedia.txt",
}


def count_tokens(sp, path):
    total_tokens = 0
    total_chars = 0
    total_bytes = 0

    print(f"\nCounting: {path.name}")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line_number, line in enumerate(f, 1):

            if not line:
                continue

            ids = sp.encode(
                line,
                out_type=int,
            )

            total_tokens += len(ids)
            total_chars += len(line)
            total_bytes += len(
                line.encode("utf-8")
            )

            if line_number % 100000 == 0:
                print(
                    f"  {line_number:,} lines | "
                    f"{total_tokens:,} tokens"
                )

    return (
        total_tokens,
        total_chars,
        total_bytes,
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

    results = {}

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        tokens, chars, bytes_ = count_tokens(
            sp,
            path,
        )

        results[name] = {
            "tokens": tokens,
            "chars": chars,
            "bytes": bytes_,
        }

        print(
            f"\n{name}"
        )
        print(
            f"  Tokens: {tokens:,}"
        )
        print(
            f"  Characters: {chars:,}"
        )
        print(
            f"  Bytes: {bytes_ / 1024 / 1024:.2f} MB"
        )

    total_tokens = sum(
        r["tokens"]
        for r in results.values()
    )

    print("\n" + "=" * 60)
    print("TOKEN STATISTICS")
    print("=" * 60)

    for name, result in results.items():

        percentage = (
            result["tokens"]
            / total_tokens
            * 100
        )

        print(
            f"{name:15s}: "
            f"{result['tokens']:>15,} tokens "
            f"({percentage:6.2f}%)"
        )

    print("-" * 60)

    print(
        f"{'TOTAL':15s}: "
        f"{total_tokens:>15,} tokens"
    )

    print("=" * 60)

    print("\nTarget mixture:")
    print("FineWeb-Edu : 75%")
    print("Code        : 15%")
    print("Wikipedia   : 10%")


if __name__ == "__main__":
    main()