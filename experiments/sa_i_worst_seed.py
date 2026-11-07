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


def acceptance_rates(result):
    records = result["worsening_moves"]
    window = max(1, (result["evals"] - 1) // 10)

    early = [
        accepted
        for evaluation, _, accepted in records
        if 2 <= evaluation <= window + 1
    ]

    late = [
        accepted
        for evaluation, _, accepted in records
        if result["evals"] - window < evaluation <= result["evals"]
    ]

    early_rate = 100 * sum(early) / len(early) if early else None
    late_rate = 100 * sum(late) / len(late) if late else None

    return early_rate, late_rate


if __name__ == "__main__":
    d = cp34_instance(10, "B", 2)

    result = solve(
        d, 2, 100, 5000,
        params={"T0": 1.0, "Tend": 0.01}
    )

    transfers, swaps = count_improving_moves(d, result["solution"])
    early_rate, late_rate = acceptance_rates(result)

    print("Seed:", 100)
    print("Cost:", result["cost"])
    print("Feasible:", result["feasible"])
    print("Improving transfers:", transfers)
    print("Improving swaps:", swaps)

    if early_rate is not None:
        print("Early acceptance rate:", round(early_rate, 2), "%")
    else:
        print("Early acceptance rate: N/A")

    if late_rate is not None:
        print("Late acceptance rate:", round(late_rate, 2), "%")
    else:
        print("Late acceptance rate: N/A")