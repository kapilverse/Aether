from datasets import load_dataset

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True,
)

for i, row in enumerate(ds):
    print(row["text"][:500])
    if i == 2:
        break

print("FineWeb-Edu streaming works.")