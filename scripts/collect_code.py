from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "data" / "raw" / "code.txt"

# We deliberately collect a controlled amount of code.
# The final 75/15/10 mixture will be enforced after
# tokenization, so this is only a source pool.

TARGET_BYTES = 400 * 1024 * 1024  # 400 MB

LANGUAGES = {
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C",
    "C++",
    "C#",
    "Go",
    "Rust",
    "PHP",
    "Ruby",
    "Swift",
    "Kotlin",
    "SQL",
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

print("Loading github-top-code...")
print("Target:", TARGET_BYTES / 1024 / 1024, "MB")
print("Languages:", ", ".join(sorted(LANGUAGES)))

dataset = load_dataset(
    "ronantakizawa/github-top-code",
    split="train",
    streaming=True,
)

written = 0
files_seen = 0
files_used = 0

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    errors="ignore",
) as out:

    for example in dataset:

        files_seen += 1

        language = example.get(
            "file_language"
        )

        if language not in LANGUAGES:
            continue

        content = example.get("content")

        if not content:
            continue

        # Separate source files.
        out.write(content)
        out.write("\n\n")

        written += len(
            content.encode("utf-8")
        )

        files_used += 1

        if files_used % 1000 == 0:
            print(
                f"Used: {files_used:,} files | "
                f"Written: {written / 1024 / 1024:.1f} MB | "
                f"Seen: {files_seen:,}"
            )

        if written >= TARGET_BYTES:
            break

print("\nCollection complete.")
print(f"Files seen: {files_seen:,}")
print(f"Files used: {files_used:,}")
print(
    f"Code written: "
    f"{written / 1024 / 1024:.2f} MB"
)
print(f"Output: {OUTPUT}")