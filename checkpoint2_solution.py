"""Checkpoint 2 -- Team T10 (seed 10), Storyline B: balancing CPU load.

Run:  python checkpoint2_solution.py
Needs your Checkpoint 1 module that provides cloud_variant(seed, variant).
Edit the import in load_data() if your module has another name.
Writes results_tables.tex (included by report.tex).
"""
import numpy as np
from checkpoint2_generator_PUBLIC import checkpoint2_params, rotation, X0_QUAD

SEED = 10
TOL = 1e-6          # relative stopping rule: ||grad_k|| <= TOL * ||grad_0||
MAXIT = 20000


# ----------------------------------------------------------------- helpers
def check_grad(f, grad, x, h=1e-6):
    """max abs difference between grad and central differences."""
    x = np.asarray(x, float)
    num = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        num[i] = (f(x + e) - f(x - e)) / (2 * h)
    return np.max(np.abs(num - grad(x)))


def fd_hess(grad, x, h=1e-6):
    """central differences of a verified gradient, symmetrized."""
    x = np.asarray(x, float); n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)


# ----------------------------------------------------- quadratic / Rosenbrock
def make_quadratics(c, theta_deg):
    R = rotation(theta_deg)
    Q = np.diag([1.0, c])                      # f(x) = x1^2 + c x2^2 = x^T diag(1,c) x
    Hq = 2 * Q
    q1 = (lambda x: x @ Q @ x, lambda x: Hq @ x, lambda x: Hq)
    q2 = (lambda x: (R.T @ x) @ Q @ (R.T @ x),
          lambda x: R @ (Hq @ (R.T @ x)),
          lambda x: R @ Hq @ R.T)
    return q1, q2, R


def rosen(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosen_grad(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                     [-400 * x[0], 200.0]])


# ------------------------------------------------------------ load balancing
class LoadProblem:
    """F(z) = J(softmax with last logit 0),  J(w) = (1/N) sum_j (D w_j / C_j)^2."""

    def __init__(self, cap, req):
        self.C = np.asarray(cap, float)
        self.N = len(self.C)
        self.D = float(np.sum(req))
        self.a = self.D ** 2 / (self.N * self.C ** 2)      # J = sum_j a_j w_j^2

    def w(self, z):
        e = np.exp(np.append(z - np.max(np.append(z, 0)), -np.max(np.append(z, 0))))
        return e / e.sum()

    def J(self, w):
        return float(np.sum(self.a * w ** 2))

    def F(self, z):
        return self.J(self.w(z))

    def grad(self, z):
        # dF/dz_k = 2 w_k (a_k w_k - J),  k < N   (chain rule with dw_j/dz_k = w_j(delta_jk - w_k))
        w = self.w(z); J = self.J(w)
        return (2 * w * (self.a * w - J))[:-1]

    def hess(self, z):
        # analytic: d/dz_m [2 a_k w_k^2 - 2 w_k J],  dw_k/dz_m = w_k(delta_km - w_m),  dJ/dz_m = g_m
        w = self.w(z); J = self.J(w); g = self.grad(z)
        n = len(z); wk = w[:n]; ak = self.a[:n]
        dw = np.diag(wk) - np.outer(wk, wk)                # dw_k/dz_m
        H = (4 * ak * wk - 2 * J)[:, None] * dw - 2 * np.outer(wk, g)
        return 0.5 * (H + H.T)

    # closed form
    def w_star(self):
        return self.C ** 2 / np.sum(self.C ** 2)

    def J_star(self):
        return self.D ** 2 / (self.N * np.sum(self.C ** 2))

    def z_star(self):
        ws = self.w_star()
        return np.log(ws[:-1] / ws[-1])

    def J_prop(self):
        return (self.D / np.sum(self.C)) ** 2              # w_j ~ C_j  ->  all u_j = D / sum C


# ------------------------------------------------------------------ solvers
def backtrack(f, x, fx, g, d, a0=1.0, c1=1e-4, rho=0.5, maxit=60):
    a, slope = a0, g @ d
    for _ in range(maxit):
        if f(x + a * d) <= fx + c1 * a * slope:
            return a
        a *= rho
    return a


def run(method, f, grad, hess, x0, tol=TOL, maxit=MAXIT, alpha=1e-3, beta=0.9):
    """method in gd, momentum, adam, newton (pure), dnewton (damped, Armijo)."""
    x = np.array(x0, float); g0 = np.linalg.norm(grad(x)); it = 0
    v = np.zeros_like(x); m = np.zeros_like(x); s = np.zeros_like(x)
    while it < maxit:
        g = grad(x)
        if np.linalg.norm(g) <= tol * g0:
            break
        if method == "gd":
            x = x + backtrack(f, x, f(x), g, -g) * (-g)
        elif method == "momentum":
            v = beta * v - alpha * g; x = x + v
        elif method == "adam":
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g * g
            mh = m / (1 - 0.9 ** (it + 1)); sh = s / (1 - 0.999 ** (it + 1))
            x = x - alpha * mh / (np.sqrt(sh) + 1e-8)
        else:
            H = hess(x)
            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g
            if method == "dnewton":
                if g @ p >= 0:                               # H not positive definite -> shift H + tau I
                    lam_min = np.linalg.eigvalsh(H)[0]
                    p = -np.linalg.solve(H + (abs(lam_min) + 1e-3 * np.linalg.norm(g)) * np.eye(len(x)), g)
                x = x + backtrack(f, x, f(x), g, p) * p
            else:
                x = x + p
        it += 1
    return x, it, np.linalg.norm(grad(x)) <= tol * g0


# --------------------------------------------------------------------- data
def load_data():
    try:
        from checkpoint1 import cloud_variant            # <-- your Checkpoint 1 module
        d = cloud_variant(10, 2)
        return np.asarray(d["cpu_cap"]), np.asarray(d["cpu_req"]), False
    except Exception as err:                              # demo data so the script still runs
        print("!! cloud_variant not found (%s) -> DEMO DATA, numbers are NOT yours" % err)
        rng = np.random.default_rng(0)
        return rng.integers(8, 40, 11).astype(float), rng.integers(1, 9, 50).astype(float), True


def main():
    p = checkpoint2_params(SEED)
    print(p)
    c, th = p["c"], p["theta_deg"]
    q1, q2, R = make_quadratics(c, th)
    probs = {"Q1": (q1, X0_QUAD), "Q2": (q2, R @ X0_QUAD),
             "R1": ((rosen, rosen_grad, rosen_hess), np.array([-1.2, 1.0])),
             "R2": ((rosen, rosen_grad, rosen_hess), p["x0_rosen"])}
    print("\n-- check values (expected: 24.2/232.86, 2.75106/20.5005, 93.32/261.813)")
    for name, ((f, g, H), x0) in probs.items():
        print(name, f(x0), np.linalg.norm(g(x0)), " check_grad err", check_grad(f, g, x0))

    rows = []
    print("\n-- solver comparison on Q1,Q2,R1,R2 (iterations, converged)")
    for name, ((f, g, H), x0) in probs.items():
        for meth in ["gd", "momentum", "adam", "newton", "dnewton"]:
            x, it, ok = run(meth, f, g, H, x0, alpha=1e-3 if name[0] == "R" else 1e-2)
            print("%-3s %-9s it=%6d ok=%s  f=%.3e" % (name, meth, it, ok, f(x)))

    cap, req, demo = load_data()
    P = LoadProblem(cap, req)
    z0 = np.zeros(P.N - 1)
    print("\nN=%d  D=%g   F(z0)=%.6f (expect 0.237586)  |gradF(z0)|=%.7f (expect 0.0504731)"
          % (P.N, P.D, P.F(z0), np.linalg.norm(P.grad(z0))))
    print("check_grad err at z0:", check_grad(P.F, P.grad, z0),
          " | analytic vs FD Hessian:", np.max(np.abs(P.hess(z0) - fd_hess(P.grad, z0))))
    zs = P.z_star()
    print("|gradF(z*)|_inf =", np.max(np.abs(P.grad(zs))), "  F(z*)-J* =", P.F(zs) - P.J_star())

    lines = []
    for meth in ["gd", "momentum", "adam", "newton", "dnewton"]:
        z, it, ok = run(meth, P.F, P.grad, P.hess, z0, alpha=1e-1)
        w = P.w(z); ev = np.linalg.eigvalsh(P.hess(z))
        err_w, err_J = np.max(np.abs(w - P.w_star())), abs(P.J(w) - P.J_star())
        print("%-9s it=%5d ok=%s  |w-w*|inf=%.2e  |J-J*|=%.2e  eig[min,max]=[%.3e, %.3e]"
              % (meth, it, ok, err_w, err_J, ev[0], ev[-1]))
        lines.append("%s & %d & %.2e & %.2e & %.2e & %.2e \\\\" % (meth, it, err_w, err_J, ev[0], ev[-1]))
    with open("results_tables.tex", "w") as fh:
        fh.write("%% generated by checkpoint2_solution.py%s\n" % ("  (DEMO DATA!)" if demo else ""))
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\nsolver & iters & $\\|w-w^*\\|_\\infty$ & $|J-J^*|$ & "
                 "$\\lambda_{\\min}(\\nabla^2F)$ & $\\lambda_{\\max}(\\nabla^2F)$ \\\\\\midrule\n")
        fh.write("\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
        fh.write("\n\\medskip\n\\noindent $J^*=%.6g$, \\quad $J_{\\rm prop}=%.6g$ (capacity-proportional split).\n"
                 % (P.J_star(), P.J_prop()))
    print("J* = %.6g   J(prop) = %.6g" % (P.J_star(), P.J_prop()))


if __name__ == "__main__":
    main()
