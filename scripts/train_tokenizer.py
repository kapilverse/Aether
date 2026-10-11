from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent

sample_dir = ROOT / "data" / "tokenizer_sample"
tokenizer_dir = ROOT / "tokenizer"

sample_dir.mkdir(parents=True, exist_ok=True)
tokenizer_dir.mkdir(parents=True, exist_ok=True)

fineweb = ROOT / "data" / "raw" / "fineweb_edu.txt"
wiki = ROOT / "data" / "raw" / "wikipedia.txt"
sample = sample_dir / "tokenizer_sample.txt"

if sample.exists():
    sample.unlink()


def copy_bytes(src, dst, target):
    written = 0

    with open(src, "rb") as fin, open(dst, "ab") as fout:
        while written < target:
            chunk = fin.read(min(1024 * 1024, target - written))

            if not chunk:
                break

            fout.write(chunk)
            written += len(chunk)

    return written


# ~225 MB FineWeb-Edu
fw = copy_bytes(
    fineweb,
    sample,
    225 * 1024 * 1024
)

# ~45 MB Wikipedia
wiki_bytes = copy_bytes(
    wiki,
    sample,
    45 * 1024 * 1024
)

print(f"FineWeb sample: {fw / 1024 / 1024:.1f} MB")
print(f"Wikipedia sample: {wiki_bytes / 1024 / 1024:.1f} MB")
print(f"Total: {(fw + wiki_bytes) / 1024 / 1024:.1f} MB")

spm.SentencePieceTrainer.train(
    input=str(sample),

    model_prefix=str(tokenizer_dir / "aether"),

    vocab_size=32000,
    model_type="bpe",

    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,

    character_coverage=1.0,
    max_sentence_length=16384,

    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
)

print("\nTokenizer training complete.")
print("Created:")
print(tokenizer_dir / "aether.model")
print(tokenizer_dir / "aether.vocab")