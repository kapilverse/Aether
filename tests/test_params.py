from model import AETHER100M, count_parameters


model = AETHER100M()

params = count_parameters(model)

print(f"Parameters: {params:,}")
print(f"Expected:   {100_092_672:,}")

assert params == 100_092_672, (
    f"Parameter mismatch: {params:,}"
)

assert model.lm_head.weight is model.embed.weight

print("Parameter count: PASS")
print("Weight tying:    PASS")