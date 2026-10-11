#!/usr/bin/env python3
"""Prepare AETHER-100M SFT3 data from four public Hugging Face datasets.

Outputs:
  sft_round3_12000.jsonl
  reports/sft_round3_manifest.json
  reports/sft_round3_preview.txt

This script only prepares data. It does not start model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SEED = 20261011
TOKENIZER_PATH = Path("tokenizer/aether.model")
OUT_PATH = Path("sft_round3_12000.jsonl")
MANIFEST_PATH = Path("reports/sft_round3_manifest.json")
PREVIEW_PATH = Path("reports/sft_round3_preview.txt")

SOURCES = [
    {
        "key": "glaive_code",
        "repo": "glaiveai/glaive-code-assistant-v3",
        "config": None,
        "split": "train",
        "license": "apache-2.0 (dataset card metadata; review source terms before redistribution)",
        "target": 5000,
        "category": "coding_technical",
        "url": "https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3",
    },
    {
        "key": "dolly",
        "repo": "databricks/databricks-dolly-15k",
        "config": None,
        "split": "train",
        "license": "CC BY-SA 3.0 (attribution/share-alike terms apply)",
        "target": 4000,
        "category": "general_instruction",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
    },
    {
        "key": "orca_math",
        "repo": "microsoft/orca-math-word-problems-200k",
        "config": None,
        "split": "train",
        "license": "MIT (dataset card metadata; review source terms before redistribution)",
        "target": 2000,
        "category": "math_reasoning",
        "url": "https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k",
    },
    {
        "key": "gsm8k",
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "license": "MIT (dataset card metadata; review source terms before redistribution)",
        "target": 1000,
        "category": "math_word_problems",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
    },
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Preserve code and line breaks but remove excessive blank lines and trailing spaces.
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    value = re.sub(r"\n{4,}", "\n\n\n", value).strip()
    return value


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def sha(text: str) -> str:
    return hashlib.sha256(norm_key(text).encode("utf-8", errors="ignore")).hexdigest()


def get_field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def normalize_row(key: str, row: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return user prompt, assistant response, fine-grained category."""
    if key == "glaive_code":
        prompt = get_field(row, "question", "prompt", "instruction")
        answer = get_field(row, "answer", "response", "output")
        category = "coding_technical"
    elif key == "dolly":
        instruction = get_field(row, "instruction", "prompt", "question")
        context = get_field(row, "context")
        answer = get_field(row, "response", "answer", "output")
        if context:
            prompt = f"Context:\n{context}\n\nInstruction:\n{instruction}"
        else:
            prompt = instruction
        category = "general_instruction"
    elif key == "orca_math":
        prompt = get_field(row, "question", "instruction", "prompt")
        answer = get_field(row, "answer", "response", "output")
        category = "math_reasoning"
    elif key == "gsm8k":
        prompt = get_field(row, "question", "instruction", "prompt")
        answer = get_field(row, "answer", "response", "output")
        category = "math_word_problems"
    else:
        return None

    if not prompt or not answer:
        return None
    return prompt, answer, category


def collect_existing_prompt_hashes(root: Path) -> tuple[set[str], dict[str, int]]:
    """Collect exact prompt hashes from SFT2 JSONL and SFT1 HF dataset if available."""
    hashes: set[str] = set()
    stats: dict[str, int] = {}

    sft2_files = [root / "sft_round2_5000.jsonl", root / "sft_round2_5000.jsonl"]
    sft2_count = 0
    for path in dict.fromkeys(sft2_files):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for msg in record.get("messages", []) or []:
                    if isinstance(msg, dict) and str(msg.get("role", "")).lower() in {"user", "human"}:
                        prompt = clean_text(msg.get("content", ""))
                        if prompt:
                            hashes.add(sha(prompt))
                            sft2_count += 1
                            break
    stats["sft2_prompts_loaded"] = sft2_count

    sft1_dir = root / "sft" / "data" / "ultrachat_20k"
    sft1_count = 0
    if sft1_dir.exists():
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(sft1_dir))
            # Dataset splits can be either a Dataset or DatasetDict.
            if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
                split = next(iter(ds.keys()), None)
                ds = ds[split] if split else None
            if ds is not None:
                for row in ds:
                    found = False
                    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", msg.get("from", ""))).lower()
                            if role in {"user", "human"}:
                                prompt = clean_text(msg.get("content", msg.get("value", "")))
                                if prompt:
                                    hashes.add(sha(prompt))
                                    sft1_count += 1
                                    found = True
                                    break
                    if not found:
                        prompt = get_field(row, "prompt", "instruction", "question")
                        if prompt:
                            hashes.add(sha(prompt))
                            sft1_count += 1
        except Exception as exc:
            print(f"WARNING: Could not inspect SFT1 dataset at {sft1_dir}: {exc}", file=sys.stderr)
    else:
        print(f"NOTE: SFT1 dataset not found at {sft1_dir}; SFT1 overlap check will be skipped.", file=sys.stderr)
    stats["sft1_prompts_loaded"] = sft1_count
    stats["unique_existing_prompt_hashes"] = len(hashes)
    return hashes, stats


def simple_quality_check(prompt: str, answer: str, sp, max_tokens: int = 2048) -> tuple[bool, str, int]:
    if len(prompt) < 12:
        return False, "prompt_too_short", 0
    if len(answer) < 12:
        return False, "answer_too_short", 0
    if len(prompt) > 12000 or len(answer) > 18000:
        return False, "character_limit", 0

    # Reject common synthetic placeholders and obvious empty response patterns.
    low = answer.casefold().strip()
    blocked = [
        "as an ai language model, i cannot",
        "insert answer here",
        "your answer here",
        "lorem ipsum",
        "i cannot answer this question because it is not provided",
    ]
    if any(x in low for x in blocked):
        return False, "placeholder_or_refusal_template", 0

    # Reject a response dominated by a single repeated non-empty line.
    lines = [x.strip() for x in answer.splitlines() if x.strip()]
    if len(lines) >= 4:
        line_counts = Counter(lines)
        if line_counts.most_common(1)[0][1] / len(lines) > 0.55:
            return False, "repeated_lines", 0

    text = f"User:\n{prompt}\n\nAssistant:\n{answer}\n\n"
    try:
        ids = sp.encode(text, out_type=int)
        token_count = len(ids)
    except TypeError:
        token_count = len(sp.encode(text))
    if token_count > max_tokens:
        return False, "over_2048_tokens", token_count
    if token_count < 12:
        return False, "too_few_tokens", token_count
    return True, "accepted", token_count


def load_streaming_dataset(source: dict[str, Any], seed: int):
    from datasets import load_dataset
    args = {"path": source["repo"], "split": source["split"], "streaming": True}
    if source["config"]:
        args["name"] = source["config"]
    ds = load_dataset(**args)
    try:
        ds = ds.shuffle(seed=seed, buffer_size=10000)
    except TypeError:
        ds = ds.shuffle(seed=seed)
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="AETHER-100M project root")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--allow-dolly", action="store_true", help="Explicitly include CC BY-SA 3.0 Dolly data")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and print source plan; do not download")
    args = parser.parse_args()
    root = args.root.resolve()

    print("AETHER-100M SFT3 dataset preparation")
    print(f"Project root: {root}")
    print(f"Seed: {args.seed}")
    print("Planned targets: " + ", ".join(f"{s['key']}={s['target']}" for s in SOURCES))
    print("Total target:", sum(s["target"] for s in SOURCES))

    tokenizer_path = root / TOKENIZER_PATH
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    except Exception as exc:
        raise SystemExit(f"Could not load SentencePiece tokenizer: {exc}")

    if args.dry_run:
        print("Tokenizer loaded successfully. Dry run complete; no datasets downloaded.")
        return 0

    if not args.allow_dolly:
        raise SystemExit(
            "Dolly is licensed CC BY-SA 3.0. Because attribution/share-alike implications need deliberate review, "
            "the script will not include it unless you explicitly pass --allow-dolly. "
            "For a portfolio/research run, inspect the license and decide whether to proceed."
        )

    try:
        from datasets import __version__ as datasets_version
    except Exception:
        datasets_version = "unknown"

    existing_hashes, existing_stats = collect_existing_prompt_hashes(root)
    # Keep a second set for duplicates between newly selected sources.
    used_prompt_hashes = set(existing_hashes)
    accepted: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, int]] = {}
    start = time.time()

    for source in SOURCES:
        key = source["key"]
        want = source["target"]
        counters: Counter[str] = Counter()
        print(f"\nLoading {source['repo']} [{source['split']}] for {want:,} examples ...", flush=True)
        try:
            stream = load_streaming_dataset(source, args.seed)
            for idx, row in enumerate(stream):
                counters["scanned"] += 1
                normalized = normalize_row(key, dict(row))
                if normalized is None:
                    counters["missing_prompt_or_answer"] += 1
                    continue
                prompt, answer, category = normalized
                p_hash = sha(prompt)
                if p_hash in used_prompt_hashes:
                    counters["duplicate_prompt"] += 1
                    continue
                ok, reason, token_count = simple_quality_check(prompt, answer, sp, args.max_tokens)
                if not ok:
                    counters[reason] += 1
                    continue
                record = {
                    "id": f"sft3-{key}-{counters['accepted']:05d}",
                    "category": category,
                    "source_dataset": source["repo"],
                    "license": source["license"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "_token_count": token_count,
                }
                accepted.append(record)
                used_prompt_hashes.add(p_hash)
                counters["accepted"] += 1
                if counters["accepted"] >= want:
                    break
                if counters["scanned"] % 5000 == 0:
                    print(f"  scanned={counters['scanned']:,}, accepted={counters['accepted']:,}/{want:,}", flush=True)
        except Exception as exc:
            counters["source_error"] += 1
            print(f"ERROR while processing {source['repo']}: {exc}", file=sys.stderr)
        source_stats[key] = dict(counters)
        print(f"Source result: accepted {counters['accepted']:,}/{want:,}; scanned {counters['scanned']:,}", flush=True)

    # Strip internal token count from the training records and sort deterministically.
    final_records = []
    for rec in accepted:
        item = dict(rec)
        item.pop("_token_count", None)
        final_records.append(item)

    # Deterministically shuffle mixture to avoid source blocks during training.
    import random
    random.Random(args.seed).shuffle(final_records)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = root / OUT_PATH
    manifest_path = root / MANIFEST_PATH
    preview_path = root / PREVIEW_PATH

    with out_path.open("w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    source_targets = {s["key"]: s["target"] for s in SOURCES}
    source_actual = {
        s["key"]: source_stats.get(s["key"], {}).get("accepted", 0)
        for s in SOURCES
    }
    complete = all(source_actual[k] >= source_targets[k] for k in source_targets)
    manifest = {
        "project": "AETHER-100M",
        "round": "SFT3",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "target_examples": sum(source_targets.values()),
        "actual_examples": len(final_records),
        "complete_target_reached": complete,
        "dataset_file": str(out_path.relative_to(root)),
        "format": "JSONL, one record per example, User/Assistant messages",
        "tokenizer": str(TOKENIZER_PATH),
        "max_sequence_tokens": args.max_tokens,
        "deduplication": "exact normalized user-prompt SHA-256 against SFT1/SFT2 where available and across SFT3",
        "existing_dataset_checks": existing_stats,
        "datasets_library_version": datasets_version,
        "source_targets": source_targets,
        "source_actual": source_actual,
        "sources": [
            {
                "key": s["key"],
                "repo": s["repo"],
                "split": s["split"],
                "license": s["license"],
                "target": s["target"],
                "url": s["url"],
                "stats": source_stats.get(s["key"], {}),
            }
            for s in SOURCES
        ],
        "notes": [
            "Training has not been started by this script.",
            "Dataset card license metadata is recorded as reported; upstream and applicable terms must be reviewed before redistribution or commercial use.",
            "Dolly is CC BY-SA 3.0 and was included only after explicit --allow-dolly opt-in.",
            "Automatic filters are heuristic and do not prove factual correctness or code correctness.",
            "Exact prompt deduplication does not detect semantic/near duplicates.",
        ],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with preview_path.open("w", encoding="utf-8") as f:
        f.write("AETHER-100M SFT3 DATASET PREVIEW\n")
        f.write(f"Examples: {len(final_records)} / {sum(source_targets.values())}\n\n")
        for rec in final_records[:30]:
            f.write(f"ID: {rec['id']} | source: {rec['source_dataset']} | category: {rec['category']}\n")
            f.write("USER: " + rec["messages"][0]["content"][:600].replace("\n", "\\n") + "\n")
            f.write("ASSISTANT: " + rec["messages"][1]["content"][:800].replace("\n", "\\n") + "\n")
            f.write("-" * 80 + "\n")

    print("\nPreparation complete.")
    print(f"Examples written: {len(final_records):,} / {sum(source_targets.values()):,}")
    print(f"Dataset:  {out_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Preview:  {preview_path}")
    print("Source counts:")
    for key in source_targets:
        print(f"  {key}: {source_actual[key]:,}/{source_targets[key]:,}")
    if not complete:
        print("WARNING: Target not reached. Do not train yet; inspect manifest/source statistics.", file=sys.stderr)
        return 2
    print("IMPORTANT: Review the preview and licensing notes before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
