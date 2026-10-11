from datasets import load_dataset
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_BYTES = 30 * 1024 * 1024

print("Collecting permissively licensed code...")

ds = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",
    split="train",
    streaming=True,
)

written = 0

with open(OUT / "code.txt", "w", encoding="utf-8") as f:
    for row in ds:
        text = row.get("content", "").strip()

        if not text:
            continue

        f.write(text)
        f.write("\n")

        written += len(text.encode("utf-8"))

        if written >= TARGET_BYTES:
            break

print(f"Code: {written / 1024 / 1024:.1f} MB")
print("Code collection complete.")