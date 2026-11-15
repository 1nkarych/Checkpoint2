import random
import math
import numpy as np
from checkpoint4_generator_PUBLIC import evaluate_B, latency_B


def initial_solution(d, rng):
    s = d["S"]
    m = d["M"]
    allowed = [
        [
            j for j in range(s)
            if d["vm_required_zone"][i] == -1
            or d["zone"][j] == d["vm_required_zone"][i]
        ]
        for i in range(m)
    ]

    for _ in range(200):
        assign = [-1] * m
        cpu = [0] * s
        mem = [0] * s
        order = list(range(m))
        rng.shuffle(order)
        order.sort(key=lambda i: (len(allowed[i]), -d["cpu_req"][i], -d["mem_req"][i]))

        for i in order:
            choices = [
                j for j in allowed[i]
                if cpu[j] + d["cpu_req"][i] <= d["cpu_cap"][j]
                and mem[j] + d["mem_req"][i] <= d["mem_cap"][j]
            ]

            if not choices:
                break

            j = min(
                choices,
                key=lambda j: (
                    latency_B((cpu[j] + d["cpu_req"][i]) / d["cpu_cap"][j])
                    - latency_B(cpu[j] / d["cpu_cap"][j]),
                    rng.random()
                )
            )

            assign[i] = j
            cpu[j] += d["cpu_req"][i]
            mem[j] += d["mem_req"][i]
        else:
            return assign

    raise ValueError("Could not construct a feasible initial solution")


def transfer_move(d, assign, rng):
    i = rng.randrange(d["M"])
    old = assign[i]
    targets = [
        j for j in range(d["S"])
        if j != old
        and (
            d["vm_required_zone"][i] == -1
            or d["zone"][j] == d["vm_required_zone"][i]
        )
    ]

    rng.shuffle(targets)

    for j in targets:
        cpu = sum(
            d["cpu_req"][k]
            for k in range(d["M"]) if assign[k] == j
        )
        mem = sum(
            d["mem_req"][k]
            for k in range(d["M"]) if assign[k] == j
        )

        if (
            cpu + d["cpu_req"][i] <= d["cpu_cap"][j]
            and mem + d["mem_req"][i] <= d["mem_cap"][j]
        ):
            neighbor = assign.copy()
            neighbor[i] = j
            return neighbor

    return None


def incremental_cost(d, assign, neighbor, cost):
    changes = [i for i in range(d["M"]) if assign[i] != neighbor[i]]

    if not changes:
        return float(cost)

    if len(changes) != 1:
        raise ValueError("Expected exactly one transfer")

    i = changes[0]
    source = assign[i]
    target = neighbor[i]

    cpu_source = sum(
        d["cpu_req"][k]
        for k in range(d["M"]) if assign[k] == source
    )
    cpu_target = sum(
        d["cpu_req"][k]
        for k in range(d["M"]) if assign[k] == target
    )
    size = d["cpu_req"][i]

    return float(
        cost
        - latency_B(cpu_source / d["cpu_cap"][source])
        - latency_B(cpu_target / d["cpu_cap"][target])
        + latency_B((cpu_source - size) / d["cpu_cap"][source])
        + latency_B((cpu_target + size) / d["cpu_cap"][target])
    )


def solve(d, variant, seed, budget, params=None, start=None):
    if budget < 1:
        raise ValueError("budget must be positive")

    rng = random.Random(seed)
    current = list(start) if start is not None else initial_solution(d, rng)
    current_cost, problems = evaluate_B(d, current)

    if problems:
        raise ValueError("Starting solution is infeasible")

    best = current.copy()
    best_cost = float(current_cost)
    evals = 1
    trace = [(1, best_cost)]
    worsening_moves = []
    positive = []

    warmup = min(3000, budget - 1) if params is None else 0

    for _ in range(warmup):
        candidate = transfer_move(d, current, rng)

        if candidate is None:
            candidate = current.copy()

        cost = incremental_cost(d, current, candidate, current_cost)
        evals += 1
        delta = cost - current_cost

        if delta > 0:
            positive.append(delta)
            worsening_moves.append((evals, None, True))

        current = candidate
        current_cost = cost

        if cost < best_cost:
            best = candidate.copy()
            best_cost = float(cost)

        if evals % 1000 == 0:
            trace.append((evals, best_cost))

    if params is None:
        if positive:
            t0 = 4.5 * float(np.median(positive))
            tend = 0.22 * float(np.percentile(positive, 5))
        else:
            t0, tend = 1.0, 0.01
    else:
        t0 = float(params["T0"])
        tend = float(params["Tend"])

    if not (math.isfinite(t0) and math.isfinite(tend)) or t0 <= 0 or tend <= 0:
        raise ValueError("T0 and Tend must be finite and positive")

    steps = budget - evals

    for k in range(steps):
        temperature = t0 * (tend / t0) ** (k / max(steps - 1, 1))
        candidate = transfer_move(d, current, rng)

        if candidate is None:
            candidate = current.copy()

        cost = incremental_cost(d, current, candidate, current_cost)
        evals += 1
        delta = cost - current_cost

        accepted = delta <= 0 or rng.random() < math.exp(-delta / temperature)

        if delta > 0:
            worsening_moves.append((evals, temperature, bool(accepted)))

        if accepted:
            current = candidate
            current_cost = cost

            if cost < best_cost:
                best = candidate.copy()
                best_cost = float(cost)

        if evals % 1000 == 0:
            trace.append((evals, best_cost))

    checked_cost, problems = evaluate_B(d, best)

    if trace[-1][0] != evals:
        trace.append((evals, best_cost))

    return {
        "solution": best,
        "cost": float(checked_cost),
        "evals": evals,
        "feasible": not problems,
        "trace": trace,
        "worsening_moves": worsening_moves,
        "params_used": {"T0": t0, "Tend": tend}
    }