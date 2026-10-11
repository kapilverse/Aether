import json
import math
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from model import AETHER100M
from training.data import create_train_loader


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "processed" / "train"
VAL_DIR = ROOT / "data" / "processed" / "val"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL / DATA CONFIG
# ============================================================

SEQ_LEN = 2048

MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8

EFFECTIVE_BATCH_TOKENS = (
    MICRO_BATCH_SIZE
    * SEQ_LEN
    * GRAD_ACCUM_STEPS
)

TOTAL_TRAIN_TOKENS = 1_800_000_000

# Number of optimizer updates required to consume
# approximately the entire training set.
TOTAL_STEPS = math.ceil(
    TOTAL_TRAIN_TOKENS
    / EFFECTIVE_BATCH_TOKENS
)


# ============================================================
# OPTIMIZATION
# ============================================================

LR = 6e-4

WEIGHT_DECAY = 0.1

BETAS = (0.9, 0.95)

GRAD_CLIP = 1.0


# ============================================================
# WSD SCHEDULER
# ============================================================

WARMUP_STEPS = 400

# Final 15% of training is decay.
DECAY_FRACTION = 0.15

DECAY_STEPS = max(
    1,
    int(TOTAL_STEPS * DECAY_FRACTION),
)

DECAY_START_STEP = (
    TOTAL_STEPS - DECAY_STEPS
)


# ============================================================
# VALIDATION / CHECKPOINTING
# ============================================================

VAL_BATCH_SIZE = 4

VAL_BATCHES = 100

VAL_EVERY = 200

SAVE_EVERY = 200

# Keep a safety checkpoint relatively frequently.
SAVE_EVERY_MINUTES = 30


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CHECKPOINT
# ============================================================

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "latest.pt"
)


def get_lr(step):
    """
    WSD schedule:

    1. Linear warmup
    2. Constant LR
    3. Linear decay to 10% of LR
    """

    if step < WARMUP_STEPS:

        return LR * (
            (step + 1)
            / WARMUP_STEPS
        )

    if step < DECAY_START_STEP:

        return LR

    decay_progress = (
        step - DECAY_START_STEP
    ) / max(
        1,
        DECAY_STEPS,
    )

    decay_progress = min(
        1.0,
        max(0.0, decay_progress),
    )

    # Decay to 10% of peak LR.
    return LR * (
        1.0 - 0.9 * decay_progress
    )


def set_learning_rate(
    optimizer,
    lr,
):

    for group in optimizer.param_groups:

        group["lr"] = lr


def save_checkpoint(
    model,
    optimizer,
    step,
    tokens_seen,
    best_val_loss,
    elapsed_seconds,
):

    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_tokens":
                EFFECTIVE_BATCH_TOKENS,
            "total_train_tokens":
                TOTAL_TRAIN_TOKENS,
            "lr": LR,
            "warmup_steps":
                WARMUP_STEPS,
            "decay_steps":
                DECAY_STEPS,
        },
    }

    temp_path = (
        CHECKPOINT_DIR / "latest.tmp.pt"
    )

    torch.save(
        checkpoint,
        temp_path,
    )

    temp_path.replace(
        LATEST_CHECKPOINT
    )

    # Keep only the rolling latest checkpoint.
    #
    # Numbered checkpoints are intentionally disabled because each
    # checkpoint is ~1.2 GB and would eventually fill the EC2 root disk.

    print(
        f"[CHECKPOINT] Saved: "
        f"{LATEST_CHECKPOINT}"
    )


def load_checkpoint(
    model,
    optimizer,
):

    if not LATEST_CHECKPOINT.exists():

        print(
            "[CHECKPOINT] No checkpoint found."
        )

        return (
            0,
            0,
            float("inf"),
            0.0,
        )

    print(
        f"[CHECKPOINT] Loading "
        f"{LATEST_CHECKPOINT}"
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    step = checkpoint["step"]

    tokens_seen = checkpoint[
        "tokens_seen"
    ]

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    elapsed_seconds = checkpoint.get(
        "elapsed_seconds",
        0.0,
    )

    print(
        f"[CHECKPOINT] Resuming from "
        f"step {step}"
    )

    print(
        f"[CHECKPOINT] Tokens seen: "
        f"{tokens_seen:,}"
    )

    return (
        step,
        tokens_seen,
        best_val_loss,
        elapsed_seconds,
    )


@torch.no_grad()
def evaluate(
    model,
    val_loader,
):

    model.eval()

    data_iter = iter(
        val_loader
    )

    total_loss = 0.0
    batches = 0

    for _ in range(VAL_BATCHES):

        input_ids, labels = next(
            data_iter
        )

        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if DEVICE == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                output = model(
                    input_ids,
                    labels=labels,
                )

        else:

            output = model(
                input_ids,
                labels=labels,
            )

        loss = output["loss"]

        total_loss += loss.item()

        batches += 1

        del output
        del loss
        del input_ids
        del labels

    model.train()

    return total_loss / batches


def write_status(
    step,
    tokens_seen,
    train_loss,
    val_loss,
    tok_per_sec,
    elapsed_seconds,
):

    status = {
        "step": step,
        "total_steps": TOTAL_STEPS,
        "tokens_seen": tokens_seen,
        "total_train_tokens":
            TOTAL_TRAIN_TOKENS,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tokens_per_second": tok_per_sec,
        "elapsed_seconds":
            elapsed_seconds,
        "effective_batch_tokens":
            EFFECTIVE_BATCH_TOKENS,
    }

    with open(
        CHECKPOINT_DIR / "status.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            status,
            f,
            indent=2,
        )


def main():

    print("=" * 70)

    print("AETHER-100M PRODUCTION TRAINING")

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"BF16: "
            f"{torch.cuda.is_bf16_supported()}"
        )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in AETHER100M().parameters()):,}"
    )

    print()

    print("TRAINING CONFIG")

    print("-" * 70)

    print(
        f"Sequence length: {SEQ_LEN}"
    )

    print(
        f"Micro batch: "
        f"{MICRO_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRAD_ACCUM_STEPS}"
    )

    print(
        f"Effective batch tokens: "
        f"{EFFECTIVE_BATCH_TOKENS:,}"
    )

    print(
        f"Training tokens: "
        f"{TOTAL_TRAIN_TOKENS:,}"
    )

    print(
        f"Optimizer steps: "
        f"{TOTAL_STEPS:,}"
    )

    print(
        f"Peak LR: {LR}"
    )

    print(
        f"Warmup: "
        f"{WARMUP_STEPS} steps"
    )

    print(
        f"Decay: "
        f"{DECAY_STEPS} steps"
    )

    print(
        f"Decay starts: "
        f"step {DECAY_START_STEP:,}"
    )

    print()

    torch.manual_seed(42)

    if DEVICE == "cuda":

        torch.cuda.manual_seed_all(42)

    model = AETHER100M()

    model.to(DEVICE)

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    (
        start_step,
        tokens_seen,
        best_val_loss,
        previous_elapsed,
    ) = load_checkpoint(
        model,
        optimizer,
    )

    if start_step >= TOTAL_STEPS:

        print()

        print(
            "Training is already complete."
        )

        return

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_loader = create_train_loader(
        data_dir=TRAIN_DIR,
        batch_size=MICRO_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_workers=0,
    )

    val_loader = create_train_loader(
        data_dir=VAL_DIR,
        batch_size=VAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_workers=0,
    )

    train_iter = iter(
        train_loader
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    training_start = time.time()

    last_save_time = time.time()

    total_tokens_this_run = 0

    last_train_loss = 0.0

    last_val_loss = None

    for step in range(
        start_step,
        TOTAL_STEPS,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0

        for micro_step in range(
            GRAD_ACCUM_STEPS
        ):

            input_ids, labels = next(
                train_iter
            )

            input_ids = input_ids.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            current_lr = get_lr(step)

            set_learning_rate(
                optimizer,
                current_lr,
            )

            if DEVICE == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    output = model(
                        input_ids,
                        labels=labels,
                    )

            else:

                output = model(
                    input_ids,
                    labels=labels,
                )

            loss = output["loss"]

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss at "
                    f"step {step}, "
                    f"micro-step {micro_step}: "
                    f"{loss}"
                )

            scaled_loss = (
                loss / GRAD_ACCUM_STEPS
            )

            scaled_loss.backward()

            accumulated_loss += (
                loss.item()
            )

            del output
            del loss
            del scaled_loss
            del input_ids
            del labels

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
        )

        optimizer.step()

        last_train_loss = (
            accumulated_loss
            / GRAD_ACCUM_STEPS
        )

        tokens_this_step = (
            EFFECTIVE_BATCH_TOKENS
        )

        tokens_seen += (
            tokens_this_step
        )

        total_tokens_this_run += (
            tokens_this_step
        )

        elapsed = (
            previous_elapsed
            + time.time()
            - training_start
        )

        tok_per_sec = (
            total_tokens_this_run
            / max(
                1.0,
                time.time()
                - training_start,
            )
        )

        completed_step = step + 1

        progress = (
            tokens_seen
            / TOTAL_TRAIN_TOKENS
        )

        remaining_tokens = max(
            0,
            TOTAL_TRAIN_TOKENS
            - tokens_seen,
        )

        eta_seconds = (
            remaining_tokens
            / max(tok_per_sec, 1.0)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            completed_step % VAL_EVERY == 0
            or completed_step == 1
        ):

            print()

            print(
                "[VALIDATION] Running..."
            )

            last_val_loss = evaluate(
                model,
                val_loader,
            )

            print(
                f"[VALIDATION] "
                f"Loss: {last_val_loss:.4f}"
            )

            if last_val_loss < best_val_loss:

                best_val_loss = (
                    last_val_loss
                )

                best_path = (
                    CHECKPOINT_DIR
                    / "best.pt"
                )

                torch.save(
                    {
                        "step":
                            completed_step,
                        "tokens_seen":
                            tokens_seen,
                        "model":
                            model.state_dict(),
                        "optimizer":
                            optimizer.state_dict(),
                        "val_loss":
                            best_val_loss,
                    },
                    best_path,
                )

                print(
                    "[CHECKPOINT] New best model saved."
                )

            print()

        # ----------------------------------------------------
        # PRINT STATUS
        # ----------------------------------------------------

        if DEVICE == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            print(
                f"Step "
                f"{completed_step:04d}/"
                f"{TOTAL_STEPS} | "
                f"Loss: "
                f"{last_train_loss:.4f} | "
                f"LR: "
                f"{current_lr:.2e} | "
                f"Grad: "
                f"{float(grad_norm):.4f} | "
                f"Tokens: "
                f"{tokens_seen:,} | "
                f"Progress: "
                f"{progress * 100:.2f}% | "
                f"Tok/s: "
                f"{tok_per_sec:.0f} | "
                f"ETA: "
                f"{eta_seconds / 3600:.2f}h | "
                f"VRAM: "
                f"{allocated:.2f}/"
                f"{reserved:.2f} GB"
            )

        else:

            print(
                f"Step "
                f"{completed_step:04d}/"
                f"{TOTAL_STEPS} | "
                f"Loss: "
                f"{last_train_loss:.4f} | "
                f"LR: "
                f"{current_lr:.2e} | "
                f"Grad: "
                f"{float(grad_norm):.4f} | "
                f"Tokens: "
                f"{tokens_seen:,} | "
                f"Progress: "
                f"{progress * 100:.2f}% | "
                f"Tok/s: "
                f"{tok_per_sec:.0f} | "
                f"ETA: "
                f"{eta_seconds / 3600:.2f}h"
            )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        should_save = (
            completed_step % SAVE_EVERY == 0
            or (
                time.time()
                - last_save_time
                >= SAVE_EVERY_MINUTES * 60
            )
        )

        if should_save:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=completed_step,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
                elapsed_seconds=elapsed,
            )

            last_save_time = time.time()

            write_status(
                step=completed_step,
                tokens_seen=tokens_seen,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                tok_per_sec=tok_per_sec,
                elapsed_seconds=elapsed,
            )

    # --------------------------------------------------------
    # FINAL CHECKPOINT
    # --------------------------------------------------------

    elapsed = (
        previous_elapsed
        + time.time()
        - training_start
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=TOTAL_STEPS,
        tokens_seen=tokens_seen,
        best_val_loss=best_val_loss,
        elapsed_seconds=elapsed,
    )

    write_status(
        step=TOTAL_STEPS,
        tokens_seen=tokens_seen,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        tok_per_sec=tok_per_sec,
        elapsed_seconds=elapsed,
    )

    print()

    print("=" * 70)

    print("AETHER-100M TRAINING COMPLETE")

    print("=" * 70)

    print(
        f"Tokens trained: "
        f"{tokens_seen:,}"
    )

    print(
        f"Final train loss: "
        f"{last_train_loss:.4f}"
    )

    if last_val_loss is not None:

        print(
            f"Final validation loss: "
            f"{last_val_loss:.4f}"
        )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training time: "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Average throughput: "
        f"{tok_per_sec:.0f} tokens/sec"
    )

    print(
        f"Checkpoint: "
        f"{LATEST_CHECKPOINT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()