from datasets import load_dataset
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)


def collect_fineweb(target_bytes=2_000_000_000):
    print("Collecting FineWeb-Edu...")
    
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    written = 0

    with open(OUT / "fineweb_edu.txt", "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()

            if not text:
                continue

            f.write(text)
            f.write("\n")

            written += len(text.encode("utf-8"))

            if written >= target_bytes:
                break

    print(f"FineWeb-Edu: {written / 1e9:.2f} GB")


def collect_wikipedia(target_bytes=300_000_000):
    print("Collecting Wikipedia...")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    written = 0

    with open(OUT / "wikipedia.txt", "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()

            if not text:
                continue

            f.write(text)
            f.write("\n")

            written += len(text.encode("utf-8"))

            if written >= target_bytes:
                break

    print(f"Wikipedia: {written / 1e6:.0f} MB")


if __name__ == "__main__":
    collect_fineweb()
    collect_wikipedia()
    print("Data collection complete.")