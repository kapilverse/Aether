from pathlib import Path

import sentencepiece as spm


ROOT = Path(__file__).resolve().parent.parent

TOKENIZER = ROOT / "tokenizer" / "aether.model"
INPUT = ROOT / "data" / "raw" / "fineweb_edu.txt"


def main():
    print("Loading tokenizer...")
    
    sp = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER)
    )

    print("Vocab size:", sp.vocab_size())
    print("UNK id:    ", sp.unk_id())
    print("BOS id:    ", sp.bos_id())
    print("EOS id:    ", sp.eos_id())
    print("PAD id:    ", sp.pad_id())

    assert sp.vocab_size() == 32000
    assert sp.unk_id() == 0
    assert sp.bos_id() == 1
    assert sp.eos_id() == 2
    assert sp.pad_id() == 3

    # Read only a small amount for the test.
    with open(
        INPUT,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        text = f.read(100_000)

    ids = sp.encode(
        text,
        out_type=int,
    )

    print("\nCharacters:", len(text))
    print("Tokens:    ", len(ids))

    print("\nFirst 30 tokens:")
    print(ids[:30])

    decoded = sp.decode(ids[:30])

    print("\nDecoded:")
    print(decoded[:500])

    assert len(ids) > 0
    assert all(
        0 <= token_id < 32000
        for token_id in ids
    )

    print("\nTokenizer test: PASS")


if __name__ == "__main__":
    main()