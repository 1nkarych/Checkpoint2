
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta.ga import solve
from src.meta.interface import check
COLS = ['team', 'role', 'configuration', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'params']
SETTINGS = [{"repair": True}, {"repair": False, "rho": 1.0}, {"repair": False, "rho": 10.0}, {"repair": False, "rho": 100.0}]
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results/abdimazhit_exp.csv'); ap.add_argument('--budget', type=int, default=100000)
ap.add_argument('--setting', type=int, default=None, help='run only SETTINGS[i] (for parallel runs; then use --append)')
ap.add_argument('--append', action='store_true')
a = ap.parse_args()
d = gen.cp34_instance(10, 'B', 2)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
new = not (a.append and os.path.exists(a.out))
with open(a.out, 'a' if a.append else 'w', newline='') as fh:
    w = csv.writer(fh)
    if new: w.writerow(COLS)
    for i, params in enumerate(SETTINGS):
        if a.setting is not None and i != a.setting: continue
        for seed in range(100, 105):
            t0 = time.time(); res = solve(d, 2, seed, a.budget, params=params); sec = time.time() - t0
            cost, problems = check(d, 2, res['solution'])
            p = dict(params); p['share_feasible_offspring'] = round(res['offspring_feasible_share'], 6)
            w.writerow(['T10', 'GA-P', 'GA, repair (exp)' if params['repair'] else 'GA, penalty (exp)', seed, a.budget,
                        f'{cost:.9f}', res['evals'], len(problems) == 0, f'{sec:.2f}', json.dumps(p)])
            fh.flush(); print(params, seed, f'{cost:.6f}', len(problems) == 0, p['share_feasible_offspring'])
