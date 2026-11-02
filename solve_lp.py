
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from src.model.model_builder import build_model

model = build_model()

c = model["c"]
number_y = model["data"]["M"] * model["data"]["S"]

# y variables must be integer (binary through bounds 0 and 1).
# t variables remain continuous.
integrality = np.zeros(len(c), dtype=int)
integrality[:number_y] = 1

lower_bounds = np.array([b[0] for b in model["bounds"]])
upper_bounds = np.array([
    np.inf if b[1] is None else b[1]
    for b in model["bounds"]
])

constraints = [
    LinearConstraint(
        model["A_eq"],
        model["b_eq"],
        model["b_eq"]
    ),
    LinearConstraint(
        model["A_ub"],
        -np.inf,
        model["b_ub"]
    )
]

result = milp(
    c=c,
    integrality=integrality,
    bounds=Bounds(lower_bounds, upper_bounds),
    constraints=constraints
)

print("\n--- MILP Solver Results ---")
print("Status:", result.message)

if result.success:
    print(f"Optimal objective value: {result.fun:.6f}")

    S = model["data"]["S"]
    M = model["data"]["M"]

    print("\n--- VM assignments ---")
    for k in range(M):
        for l in range(S):
            value = result.x[model["y_index"](k, l)]
            if value > 0.5:
                print(f"VM {k} -> Server {l}")

    print("\n--- Server latency variables ---")
    for l in range(S):
        value = result.x[model["t_index"](l)]
        print(f"Server {l}: t = {value:.6f}")
else:
    print("Solver did not find an optimal solution.")
    print("Status code:", result.status)
