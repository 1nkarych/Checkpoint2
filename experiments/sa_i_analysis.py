import numpy as np
import matplotlib.pyplot as plt
from checkpoint4_generator_PUBLIC import cp34_instance
from src.meta.sa import solve

d = cp34_instance(10, "B", 2)
result = solve(d, 2, 100, 5000, params={"T0": 1.0, "Tend": 0.01})

moves = [
    (temperature, accepted)
    for _, temperature, accepted in result["worsening_moves"]
    if temperature is not None
]

print("Cost:", result["cost"])
print("Evaluations:", result["evals"])
print("Feasible:", result["feasible"])

if moves:
    temperatures = np.array([t for t, _ in moves])
    accepted = np.array([int(a) for _, a in moves])

    edges = np.geomspace(temperatures.min(), temperatures.max(), 13)
    centers = []
    rates = []

    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (temperatures >= edges[i]) & (temperatures <= edges[i + 1])
        else:
            mask = (temperatures >= edges[i]) & (temperatures < edges[i + 1])

        if np.any(mask):
            centers.append(np.sqrt(edges[i] * edges[i + 1]))
            rates.append(100 * np.mean(accepted[mask]))

    plt.plot(centers, rates, marker="o")
    plt.xscale("log")
    plt.ylim(0, 100)
    plt.xlabel("Temperature")
    plt.ylabel("Acceptance Rate (%)")
    plt.title("SA-I Worsening Move Acceptance Rate")
    plt.tight_layout()
    plt.show()