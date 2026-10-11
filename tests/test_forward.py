import torch

from model import AETHER100M


def main():
    torch.manual_seed(42)

    model = AETHER100M()
    model.eval()

    # Small test batch
    input_ids = torch.randint(
        0,
        32000,
        (2, 16),
    )

    with torch.no_grad():
        output = model(input_ids)

    logits = output["logits"]

    print("Input shape: ", tuple(input_ids.shape))
    print("Logits shape:", tuple(logits.shape))

    assert logits.shape == (
        2,
        16,
        32000,
    )

    assert torch.isfinite(logits).all()

    print("Forward pass: PASS")
    print("Logits finite: PASS")

    # --------------------------------------------------
    # Causality test
    # Changing a future token must NOT change
    # logits at earlier positions.
    # --------------------------------------------------

    input_a = input_ids.clone()
    input_b = input_ids.clone()

    # Change only the final token.
    input_b[:, -1] = (
        input_b[:, -1] + 1
    ) % 32000

    with torch.no_grad():
        logits_a = model(input_a)["logits"]
        logits_b = model(input_b)["logits"]

    # Positions 0 through T-2 must be identical.
    max_difference = (
        logits_a[:, :-1] - logits_b[:, :-1]
    ).abs().max().item()

    print(
        "Max difference before changed token:",
        max_difference,
    )

    assert max_difference < 1e-5

    print("Causality test: PASS")


if __name__ == "__main__":
    main()