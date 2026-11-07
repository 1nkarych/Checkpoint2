from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta.sa import solve


def count_improving_moves(d, assign):
    base_cost, problems = evaluate_B(d, assign)

    if problems:
        raise ValueError("Solution is infeasible")

    transfers = 0
    swaps = 0

    for i in range(d["M"]):
        for j in range(d["S"]):
            if assign[i] == j:
                continue

            candidate = assign.copy()
            candidate[i] = j

            cost, problems = evaluate_B(d, candidate)

            if not problems and cost < base_cost - 1e-9:
                transfers += 1

    for i in range(d["M"]):
        for j in range(i + 1, d["M"]):
            if assign[i] == assign[j]:
                continue

            candidate = assign.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]

            cost, problems = evaluate_B(d, candidate)

            if not problems and cost < base_cost - 1e-9:
                swaps += 1

    return transfers, swaps


if __name__ == "__main__":
    d = cp34_instance(10, "B", 2)

    result = solve(
        d, 2, 100, 5000,
        params={"T0": 1.0, "Tend": 0.01}
    )

    transfers, swaps = count_improving_moves(d, result["solution"])

    print("Cost:", result["cost"])
    print("Improving transfers:", transfers)
    print("Improving swaps:", swaps)