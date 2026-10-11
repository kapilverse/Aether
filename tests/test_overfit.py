import torch
import torch.nn.functional as F

from model import AETHER100M


def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    model = AETHER100M().to(device)
    model.train()

    # Tiny fixed dataset.
    # The model should memorize this.
    batch_size = 2
    seq_len = 32

    torch.manual_seed(123)

    input_ids = torch.randint(
        0,
        32000,
        (batch_size, seq_len),
        device=device,
    )

    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    print("\nStarting overfit test...")

    initial_loss = None

    for step in range(100):

        optimizer.zero_grad(set_to_none=True)

        output = model(
            input_ids,
            labels=labels,
        )

        loss = output["loss"]

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if step % 10 == 0 or step == 99:
            print(
                f"Step {step:3d} | "
                f"Loss: {loss.item():.4f}"
            )

    final_loss = loss.item()

    print("\nInitial loss:", initial_loss)
    print("Final loss:  ", final_loss)

    # We mainly care that loss decreases substantially.
    assert final_loss < initial_loss * 0.5, (
        "Loss did not decrease enough."
    )

    print("\nTiny overfit test: PASS")


if __name__ == "__main__":
    main()