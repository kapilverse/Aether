# AETHER-100M

**AETHER-100M** is an experimental, approximately 100-million-parameter decoder-only language model built and pretrained from scratch, then instruction-fine-tuned through multiple supervised fine-tuning (SFT) experiments.

The project covers the main stages of a small language model workflow: tokenizer use, transformer implementation, pretraining, LoRA-based supervised fine-tuning, adapter merging, evaluation, generation testing, checkpoint conversion, and preparation for publication on Hugging Face.

> **Status:** The training and fine-tuning pipelines ran successfully, and the model weights are available. However, generation quality remains a known limitation: outputs can be repetitive, incoherent, or incorrect. AETHER-100M is an educational/research artifact, not a dependable production assistant.

## Table of contents

- [Project goals](#project-goals)
- [Model overview](#model-overview)
- [Architecture](#architecture)
- [Tokenizer and text format](#tokenizer-and-text-format)
- [Pretraining](#pretraining)
- [Supervised fine-tuning](#supervised-fine-tuning)
- [Evaluation](#evaluation)
- [Generation testing and known issues](#generation-testing-and-known-issues)
- [Checkpoints and artifacts](#checkpoints-and-artifacts)
- [Using the model](#using-the-model)
- [Publishing on Hugging Face](#publishing-on-hugging-face)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [License and dataset attribution](#license-and-dataset-attribution)

## Project goals

1. Implement a decoder-only Transformer architecture.
2. Use a SentencePiece tokenizer to encode text.
3. Pretrain a model on approximately 1.8 billion tokens.
4. Fine-tune the pretrained model on instruction data using LoRA.
5. Merge trained LoRA adapters into standalone model checkpoints.
6. Compare checkpoints using held-out next-token prediction metrics.
7. Inspect generated answers and investigate EOS/stopping behavior.
8. Prepare the model, architecture source, tokenizer, configuration, and documentation for sharing.

The experiments show that supervised fine-tuning substantially improved held-out token prediction relative to the pretrained base model. They also show that lower held-out loss does not necessarily translate into coherent, useful generated responses.

## Model overview

| Property | Value |
|---|---|
| Model name | AETHER-100M |
| Model type | Decoder-only causal language model |
| Parameter count | 100,092,672 |
| Pretraining tokens | 1,800,011,776 (approximately 1.8B) |
| Vocabulary size | 32,000 |
| Maximum context length | 2,048 tokens |
| Tokenizer | SentencePiece |
| Positional encoding | Rotary positional embeddings (RoPE) |
| Attention | Grouped-query attention (GQA) |
| Normalization | RMSNorm |
| Feed-forward network | SwiGLU |
| Input/output embeddings | Tied |
| Intended use | Educational experimentation and research |

## Architecture

The implemented configuration uses:

- **12 Transformer layers**
- **Hidden size:** 768
- **Query attention heads:** 12
- **Key/value attention heads:** 4
- **Head dimension:** 64
- **Feed-forward intermediate size:** 2,048
- **Vocabulary:** 32,000 tokens
- **Context length:** 2,048 tokens
- **RMSNorm epsilon:** `1e-6`
- **Dropout:** `0`
- **Linear biases:** disabled
- **RoPE:** rotary positional embeddings
- **GQA:** grouped-query attention
- **SwiGLU:** gated feed-forward network
- **Weight tying:** token embeddings and output projection share weights

The model uses causal attention: each position is trained to predict the next token, without attending to future tokens.

### Important compatibility note

AETHER-100M uses a custom model implementation. A checkpoint file and `config.json` do not, by themselves, make it a drop-in `transformers.AutoModelForCausalLM` model. The compatible architecture implementation and loading code are required unless a dedicated Transformers integration is implemented and tested.

## Tokenizer and text format

The tokenizer model is stored as `tokenizer/aether.model`.

| Token | ID |
|---|---:|
| Unknown (`<unk>`) | 0 |
| Beginning of sequence (`<s>`) | 1 |
| End of sequence (`</s>`) | 2 |
| Padding (`<pad>`) | 3 |

The instruction fine-tuning format is plain text with explicit role labels:

```text
User:
The user's message

Assistant:
The assistant's response

```

System messages, when present, use the same pattern with a `System:` label. During SFT, loss is applied to assistant tokens, while user/system tokens are masked with label `-100`. The training objective uses the causal next-token shift.

The original SFT1/SFT2 pipeline did not append the tokenizer's EOS token ID to assistant responses. A separate EOS-supervision experiment was run later to test this behavior.

## Pretraining

The base model was pretrained from scratch before instruction fine-tuning.

| Metric | Result |
|---|---:|
| Tokens processed | 1,800,011,776 |
| Training duration | 11.31 hours |
| Average throughput | 20,379 tokens/second |
| Training steps | 13,733 |
| Best validation loss | 4.31079396 |
| Best checkpoint step | 13,600 |

The reported best validation checkpoint is `checkpoints/best.pt`; `checkpoints/latest.pt` is the rolling latest checkpoint.

### Base-model benchmark results

| Benchmark | Result |
|---|---:|
| ARC-Easy | 39.18% (931 / 2,376) |
| HellaSwag | 26.59% |

Benchmark results depend on the evaluation setup and should not be compared directly with published numbers unless the same protocol is used.

## Supervised fine-tuning

The fine-tuning experiments used Low-Rank Adaptation (LoRA), then merged trained adapters into full model checkpoints.

### Shared LoRA configuration

| Setting | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 983,040 |
| Approximate share of all parameters | 0.98% |

### SFT1: UltraChat

**Purpose:** establish instruction-following behavior on conversational instruction data.

- Source dataset: `HuggingFaceH4/ultrachat_200k`
- Local subset used: 20,000 examples
- Training: one epoch
- Starting checkpoint: pretrained base model
- Output adapter: final LoRA adapter from the first SFT run
- Merged checkpoint: `sft/merged/aether_100m_sft.pt`

SFT1 showed a large improvement in held-out assistant-token prediction compared with the base model. Generation tests nevertheless showed repetition and occasional incoherence.

### SFT2: mixed instruction data

**Purpose:** broaden instruction, coding, technical, and constrained-format coverage.

The final dataset contained **4,844 examples**, not the originally planned 5,000. The OpenAssistant source contributed fewer examples than expected; the project proceeded with the available data rather than performing additional backfilling.

| Category | Examples |
|---|---:|
| General | 2,000 |
| Python/coding | 1,500 |
| Technical | 594 |
| Format/constrained instructions | 750 |
| **Total** | **4,844** |

Reported source counts:

| Source | Examples |
|---|---:|
| `magpie_general` | 1,200 |
| `oasst2_general` | 800 |
| `self_oss_instruct` | 700 |
| `codefeedback_py` | 500 |
| `codefeedback_sql` | 300 |
| `magpie_tech` | 300 |
| `codefeedback_tech` | 250 |
| `oasst2_tech` | 44 |
| `smol_constraints` | 350 |
| `tulu3_if` | 300 |
| `smol_rewrite` | 100 |
| **Total** | **4,844** |

Training details:

- Starting checkpoint: merged SFT1 (`sft/merged/aether_100m_sft.pt`)
- One epoch
- 303 optimizer steps
- LoRA rank 16, alpha 32, dropout 0.05
- Adapter: `sft/round2_checkpoints/final_adapter/adapter_model.pt`
- Merged checkpoint: `sft/merged/aether_100m_sft2.pt`

SFT2 did not outperform SFT1 on the shared held-out evaluation. Generation tests also showed repetition, including repeated words and phrases.

### SFT3: coding-and-math-oriented data

**Purpose:** test whether additional code, technical, and mathematical examples would improve targeted responses.

The cleaned dataset contained **12,000 examples**:

| Source | Examples |
|---|---:|
| Glaive code assistant | 5,000 |
| Dolly | 4,000 |
| Orca Math | 2,000 |
| GSM8K | 1,000 |
| **Total** | **12,000** |

Dataset cleaning/audit reported:

- 12,000 records
- 0 invalid JSON records
- 0 schema errors
- 0 empty prompts or answers
- 0 duplicate prompts
- 1 repeated answer
- 0 examples exceeding the 2,048-token limit
- 0 examples below 12 tokens
- 0 remaining GSM8K `<<...>>` or `####` markers after cleanup
- Mean length: 336.8 tokens
- Maximum length: 2,036 tokens

The GSM8K cleanup replaced inline `<<...>>` calculation markers with equation text and converted `####` answer markers into a readable `Final answer:` form. Structural cleanliness does not guarantee semantic correctness or dataset quality.

Training details:

- Starting checkpoint: merged SFT2 (`sft/merged/aether_100m_sft2.pt`)
- One epoch
- 6,000 micro-batches
- 750 optimizer steps
- Micro-batch size: 2
- Gradient accumulation: 8
- Learning rate: `1e-4`
- Maximum sequence length: 2,048
- Adapter: `sft/round3_checkpoints/final_adapter/adapter_model.pt`
- Merged checkpoint: `sft/merged/aether_100m_sft3.pt`

SFT3 did not improve the shared held-out metrics over SFT1. Some generated code had a more recognizable structure, but code and prose outputs remained unreliable and repetitive.

### EOS-supervision experiment

**Motivation:** the original SFT1 training code did not append the tokenizer's EOS ID to assistant messages. Encoding the text string `</s>` is not equivalent to explicitly appending the special token ID.

A separate experiment was run:

- Starting checkpoint: merged SFT1 (`sft/merged/aether_100m_sft.pt`)
- Dataset: same 20,000-example UltraChat subset
- One epoch
- Explicitly appended `tokenizer.sp.eos_id()` to assistant labels
- LoRA configuration: rank 16, alpha 32, dropout 0.05
- Optimizer steps: 1,249
- Adapter: `sft/eos_experiment_checkpoints/final_adapter/adapter_model.pt`
- Merged checkpoint: `sft/merged/aether_100m_sft_eos.pt`

The EOS experiment produced the best recorded held-out token prediction metrics among the compared checkpoints. However, generation tests did not clearly resolve repetition or stopping behavior. In a small three-prompt comparison, neither SFT1 nor SFT1_EOS emitted EOS within the 100-token generation limit. This sample is too small to establish overall EOS behavior.

## Evaluation

A common held-out set of 500 examples was used to compare checkpoints. The evaluation included 442,453 assistant tokens. Loss and perplexity are next-token prediction metrics; token accuracy is the fraction of supervised next tokens predicted exactly.

| Checkpoint | Loss | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Base | 6.842851 | 937.1573 | 2.29% |
| SFT1 | 2.912011 | 18.3938 | 44.41% |
| SFT2 | 2.925273 | 18.6393 | 44.19% |
| SFT3 | 2.966417 | 19.4222 | 43.69% |
| SFT1_EOS | 2.8229 | 16.83 | 45.64% |

**Interpretation:**

- SFT1 substantially improved held-out next-token prediction relative to the base model.
- SFT2 and SFT3 did not improve on SFT1 under this evaluation.
- SFT1_EOS achieved the best loss, perplexity, and token accuracy in this comparison.
- These metrics do not directly measure factuality, reasoning, instruction-following quality, or usefulness of generated answers.
- Generation tests remained poor despite improved token metrics.

Evaluation artifacts reported by the project include `reports/four_checkpoint_eval.json`, `reports/four_checkpoint_eval.txt`, `reports/five_checkpoint_eval.json`, and `reports/five_checkpoint_eval.txt`.

## Generation testing and known issues

Generation comparisons were run across the base model, SFT1, SFT2, SFT3, and the EOS experiment. The test prompts included:

- Explain what a Python dictionary is.
- Write a Python function to check whether a number is prime.
- Explain the difference between a process and a thread.

The early comparison used temperature `0.8`, top-k `50`, a fixed seed, and up to 100 new tokens.

Observed issues included:

- Repeated words and phrases.
- Explanations that start plausibly but drift into incoherence.
- Incorrect or malformed code.
- Failure to stop naturally in the tested prompts.
- Occasional malformed formatting.

A later EOS probability diagnostic compared predictions after several standalone answer strings. EOS probability was higher for the EOS-trained checkpoint on most tested answers than for SFT1, but remained low. That diagnostic did not include the preceding user prompt, so it should be treated as an incomplete diagnostic rather than a definitive measure of EOS behavior.

**The central project finding is that better held-out token prediction did not yet yield reliable free-form generation.**

## Checkpoints and artifacts

| Artifact | Description |
|---|---|
| `checkpoints/best.pt` | Best base pretraining checkpoint, recorded at step 13,600 |
| `checkpoints/latest.pt` | Rolling latest base pretraining checkpoint |
| `sft/merged/aether_100m_sft.pt` | Merged SFT1 checkpoint |
| `sft/merged/aether_100m_sft2.pt` | Merged SFT2 checkpoint |
| `sft/merged/aether_100m_sft3.pt` | Merged SFT3 checkpoint |
| `sft/merged/aether_100m_sft_eos.pt` | Merged SFT1 checkpoint with explicit EOS supervision |
| `sft/round2_checkpoints/final_adapter/adapter_model.pt` | SFT2 LoRA adapter |
| `sft/round3_checkpoints/final_adapter/adapter_model.pt` | SFT3 LoRA adapter |
| `sft/eos_experiment_checkpoints/final_adapter/adapter_model.pt` | EOS experiment LoRA adapter |
| `tokenizer/aether.model` | SentencePiece tokenizer |
| `AETHER-100M-HF/model.safetensors` | Converted tensor weights for publication, if generated and verified |
| `AETHER-100M-HF/config.json` | Architecture metadata for the publication package |
| `AETHER-100M-HF/README.md` | Hugging Face model card |

The `.pt` checkpoints and `.safetensors` weights are different serialization formats. Converting between them does not change model intelligence or improve outputs. The SafeTensors file must be produced from the intended checkpoint and verified before publication; do not assume it exists unless conversion completed successfully.

## Representative repository layout

```text
SLM_100M/
├── model/
│   ├── aether.py
│   └── attention.py
├── tokenizer/
│   └── aether.model
├── checkpoints/
│   ├── best.pt
│   └── latest.pt
├── sft/
│   ├── train_sft.py
│   ├── train_sft_round2.py
│   ├── train_sft_round3.py
│   ├── train_sft_eos.py
│   └── merged/
│       ├── aether_100m_sft.pt
│       ├── aether_100m_sft2.pt
│       ├── aether_100m_sft3.pt
│       └── aether_100m_sft_eos.pt
├── scripts/
│   ├── eval_all_four.py
│   ├── eval_all_five.py
│   ├── compare_all_four_generations.py
│   └── diagnose_eos_probability.py
├── reports/
└── logs/
```

This is a representative layout based on the recorded experiments. A publication package may contain only the files needed to load and use the selected checkpoint.

## Using the model

AETHER-100M uses a custom architecture. A working inference environment requires a compatible PyTorch version, SentencePiece, the architecture implementation, the tokenizer, and the checkpoint.

The original project used PyTorch with CUDA on an NVIDIA A10G GPU. Python 3.14.4 was present in the EC2 virtual environment used during the experiments; other Python versions and environments should be tested rather than assumed compatible.

### Loading SafeTensors weights

If `model.safetensors` has been generated and verified, its tensors can be loaded using the SafeTensors library. The architecture must be instantiated first, and the state-dictionary keys must match.

```python
import torch
from safetensors.torch import load_file
from model import AETHER100M

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AETHER100M()
state_dict = load_file("model.safetensors", device="cpu")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
    )

model = model.to(device).eval()
```

This is a loading example, not a guarantee that every published folder uses exactly this path or that every environment is compatible. Confirm the actual model class import and checkpoint key names against the released source. Use `strict=True` when the checkpoint and architecture are expected to match exactly.

Load the tokenizer with:

```python
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(
    model_file="tokenizer/aether.model"
)
```

The repository should include a tested end-to-end inference script before it is advertised as ready to run.

## Publishing on Hugging Face

A useful custom-model repository should include:

```text
AETHER-100M/
├── README.md
├── config.json
├── model.safetensors
├── model/
│   └── *.py
└── tokenizer/
    └── aether.model
```

Recommended publication checklist:

1. Verify that the SafeTensors checkpoint loads and its tensor keys match the architecture.
2. Test loading the model from the extracted or cloned repository.
3. Include the tokenizer and required custom Python files.
4. State clearly that the architecture is custom and is not automatically compatible with standard Transformers auto-loading.
5. Include evaluation methodology, results, and generation limitations.
6. Verify the licenses and redistribution terms of every training dataset before assigning a license.
7. Do not publish credentials, cloud keys, private data, or unrelated machine-specific files.

## Reproducibility notes

A fully reproducible public release should record or verify:

- Exact Python and library versions.
- Exact training and evaluation commands.
- Random seeds and data preprocessing versions.
- Dataset revisions, licenses, and hashes.
- The precise pretraining corpus and its licensing/usage terms.
- The exact base checkpoint used for each fine-tuning run.
- Whether the published checkpoint was converted and verified from the intended `.pt` file.
- A tested clean-environment inference procedure.

Do not claim exact reproducibility unless these details are available and tested.

## Limitations and responsible use

AETHER-100M is a small experimental language model. It can:

- Produce repetitive or incoherent text.
- Generate incorrect factual claims.
- Produce code that fails to run.
- Fail to follow instructions consistently.
- Fail to end a response naturally.

It should not be relied on for medical, legal, financial, safety-critical, or other consequential decisions. Generated code should be reviewed and tested before use.

The project is best understood as a practical demonstration of small language model engineering and experimentation, not as a replacement for a modern production language model.

## License and dataset attribution

**No project license is assigned in this README.** Before choosing a license, verify the redistribution and attribution requirements of the model's training datasets, source code, and dependencies.

The project used or reported using data from UltraChat, OpenAssistant (OASST2), Magpie, Self-OSS-Instruct, CodeFeedback, Smol constraints/rewrite datasets, Tulu 3 instruction data, Glaive code assistant, Dolly, Orca Math, and GSM8K. This list reflects the recorded dataset sources; exact dataset revisions, licenses, and source-specific restrictions must be checked before redistribution.

If a dataset or its license cannot be verified, document the uncertainty and seek permission or omit the affected material from redistribution as appropriate.

## Summary

AETHER-100M completed a from-scratch pretraining run of approximately 1.8 billion tokens, followed by several LoRA-based supervised fine-tuning experiments. SFT1 substantially improved held-out token prediction, and explicit EOS supervision improved the recorded loss and token accuracy further. Nevertheless, free-form generation remained unreliable.

The project's value is its end-to-end engineering work and the measured lessons from the experiments: training success, improved token metrics, and useful assistant behavior are related but distinct outcomes.
