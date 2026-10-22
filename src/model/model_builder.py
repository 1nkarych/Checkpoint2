
"""M1: Matrix formulation for Team T10, Storyline B, Variant 2."""

from pathlib import Path
import sys

import numpy as np

# Make the project root available for importing the generator.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from checkpoint3_generator_PUBLIC import cp34_instance, TANGENT_POINTS


def build_model():
    """Build the LP matrix representation for the B2 instance."""

    # Load the exact instance required by Team T10.
    d = cp34_instance(10, "B", 2)

    S = d["S"]  # number of servers
    M = d["M"]  # number of virtual machines

    cpu_cap = d["cpu_cap"]
    mem_cap = d["mem_cap"]
    cpu_req = d["cpu_req"]
    mem_req = d["mem_req"]
    zone = d["zone"]
    vm_required_zone = d["vm_required_zone"]

    # Variable order:
    # y[k,l] for every VM k and server l, followed by t[l].
    number_y = M * S
    number_t = S
    number_variables = number_y + number_t

    def y_index(k, l):
        """Column index of the assignment variable y[k,l]."""
        return k * S + l

    def t_index(l):
        """Column index of the latency variable t[l]."""
        return number_y + l

    # Objective: minimize the sum of server latency variables.
    # Confirm this objective against the printed reference equation
    # in Section 2 of the assignment before submitting.
    c = np.zeros(number_variables)
    for l in range(S):
        c[t_index(l)] = 1.0

    # Bounds: y variables are between 0 and 1;
    # t variables are nonnegative and have no upper bound.
    bounds = [(0.0, 1.0) for _ in range(number_y)]
    bounds += [(0.0, None) for _ in range(number_t)]

    # Keep forbidden assignment variables, but fix them to zero.
    forbidden_pairs = []
    for k in range(M):
        for l in range(S):
            if (
                vm_required_zone[k] != -1
                and zone[l] != vm_required_zone[k]
            ):
                bounds[y_index(k, l)] = (0.0, 0.0)
                forbidden_pairs.append((k, l))

    # Equality constraints: B1, one per VM.
    A_eq = np.zeros((M, number_variables))
    b_eq = np.ones(M)
    eq_rows = []

    for k in range(M):
        for l in range(S):
            A_eq[k, y_index(k, l)] = 1.0
        eq_rows.append(f"B1_VM_{k}")

    # Inequality constraints: B2 (CPU), B3 (memory), B4 (latency).
    number_inequalities = S + S + S * len(TANGENT_POINTS)
    A_ub = np.zeros((number_inequalities, number_variables))
    b_ub = np.zeros(number_inequalities)
    ub_rows = []
    r = 0

    # B2: CPU capacity for each server.
    for l in range(S):
        for k in range(M):
            A_ub[r, y_index(k, l)] = cpu_req[k]
        b_ub[r] = cpu_cap[l]
        ub_rows.append(f"B2_CPU_server_{l}")
        r += 1

    # B3: memory capacity for each server.
    for l in range(S):
        for k in range(M):
            A_ub[r, y_index(k, l)] = mem_req[k]
        b_ub[r] = mem_cap[l]
        ub_rows.append(f"B3_MEM_server_{l}")
        r += 1

    # B4: tangent-line latency constraints.
    # g(u) = u / (1-u)
    # g'(a) = 1 / (1-a)^2
    # t[l] >= g(a) + g'(a) * (u[l] - a)
    # where u[l] = sum_k cpu_req[k]*y[k,l] / cpu_cap[l].
    for l in range(S):
        for m, a in enumerate(TANGENT_POINTS):
            slope = 1.0 / (1.0 - a) ** 2

            for k in range(M):
                A_ub[r, y_index(k, l)] = (
                    slope * cpu_req[k] / cpu_cap[l]
                )

            A_ub[r, t_index(l)] = -1.0
            b_ub[r] = slope * a - a / (1.0 - a)
            ub_rows.append(f"B4_latency_server_{l}_tangent_{m}")
            r += 1

    # Column labels make it possible to trace each matrix column.
    col_names = [
        f"y_{k}_{l}"
        for k in range(M)
        for l in range(S)
    ]
    col_names += [f"t_{l}" for l in range(S)]

    # Basic checks against the team's task file.
    assert S == 11, f"Expected 11 servers, got {S}"
    assert M == 50, f"Expected 50 VMs, got {M}"
    assert len(forbidden_pairs) == 51, (
        f"Expected 51 forbidden pairs, got {len(forbidden_pairs)}"
    )
    assert number_variables == 561
    assert len(eq_rows) + len(ub_rows) == 116

    # Verify that every forbidden assignment is fixed to zero.
    for k, l in forbidden_pairs:
        assert bounds[y_index(k, l)] == (0.0, 0.0)

    # Count all assignment variables fixed to zero.
    fixed_zero_count = sum(
        1
        for k in range(M)
        for l in range(S)
        if bounds[y_index(k, l)] == (0.0, 0.0)
    )
    assert fixed_zero_count == 51

    # Verify the assignment constraints B1.
    assert np.allclose(A_eq.sum(axis=1), S)
    assert np.allclose(b_eq, 1.0)

    print("Forbidden-assignment checks passed.")
    print("Assignment-constraint checks passed.")

    return {
        "data": d,
        "c": c,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "bounds": bounds,
        "row_names": eq_rows + ub_rows,
        "eq_row_names": eq_rows,
        "ub_row_names": ub_rows,
        "col_names": col_names,
        "forbidden_pairs": forbidden_pairs,
        "y_index": y_index,
        "t_index": t_index,
    }



if __name__ == "__main__":
    model = build_model()
    d = model["data"]

    print("Team T10 — Cloud Resource Allocation, B2")
    print(f"Servers: {d['S']}")
    print(f"Virtual machines: {d['M']}")
    print(f"Total CPU capacity: {d['cpu_cap'].sum()}")
    print(f"Total memory capacity: {d['mem_cap'].sum()}")
    print(f"Total CPU requirement: {d['cpu_req'].sum()}")
    print(f"Total memory requirement: {d['mem_req'].sum()}")
    print(f"Forbidden assignments fixed to zero: "
          f"{len(model['forbidden_pairs'])}")
    print(f"Variables: {len(model['c'])}")
    print(f"Constraint rows: "
          f"{len(model['eq_row_names']) + len(model['ub_row_names'])}")
    print(f"Equality rows: {len(model['eq_row_names'])}")
    print(f"Less-than-or-equal rows: {len(model['ub_row_names'])}")

    # Check matrix dimensions
    print("\n--- Matrix dimensions ---")
    print(f"A_eq shape: {model['A_eq'].shape}")
    print(f"b_eq shape: {model['b_eq'].shape}")
    print(f"A_ub shape: {model['A_ub'].shape}")
    print(f"b_ub shape: {model['b_ub'].shape}")
    print(f"Objective vector c shape: {model['c'].shape}")

    # Basic consistency checks
    assert model["A_eq"].shape == (50, 561)
    assert model["b_eq"].shape == (50,)
    assert model["A_ub"].shape == (66, 561)
    assert model["b_ub"].shape == (66,)
    assert model["c"].shape == (561,)

    print("\nAll matrix dimension checks passed!")
