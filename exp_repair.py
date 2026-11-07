"""GA-P experiment: repair ON vs OFF (OFF = penalty cost + rho*violation, rho in RHOS). Writes results/abdimazhit_exp.csv.
Uses solve() of GA-A (src/meta/ga.py) with params {"repair": bool, "rho": float}; solve must return the extra keys
'n_offspring' and 'n_feasible_offspring' (share of feasible offspring BEFORE repair is the interesting number; agree with GA-A).
Run (tuning/experiment seeds, never 0-9):
  python experiments/exp_repair.py --out results/abdimazhit_exp.csv
"""
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta.ga import solve
from src.meta.interface import check

COLS = ['team', 'role', 'configuration', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'params']
SETTINGS = [{"repair": True}, {"repair": False, "rho": 1.0}, {"repair": False, "rho": 10.0}, {"repair": False, "rho": 100.0}]

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results/abdimazhit_exp.csv'); ap.add_argument('--budget', type=int, default=100000)
ap.add_argument('--seeds', default='100-104')
a = ap.parse_args()
lo, hi = a.seeds.split('-'); seeds = range(int(lo), int(hi) + 1)
d = gen.cp34_instance(10, 'B', 2)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(COLS)
    for params in SETTINGS:
        for seed in seeds:
            t0 = time.time(); res = solve(d, 2, seed, a.budget, params=params, start=None); sec = time.time() - t0
            cost, problems = check(d, 2, res['solution'])
            p = dict(params); p['share_feasible_offspring'] = res['n_feasible_offspring'] / max(1, res['n_offspring'])
            w.writerow(['T10', 'GA-P', 'GA, repair (exp)', seed, a.budget, f'{cost:.9f}', res['evals'], len(problems) == 0, f'{sec:.2f}', json.dumps(p)])
            fh.flush(); print(params, seed, f'{cost:.6f}', len(problems) == 0)
