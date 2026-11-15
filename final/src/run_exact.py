"""Final: exact solver (HiGHS through scipy.optimize.milp, helper of the kit) and the LP relaxation.

  python final/src/run_exact.py --own                # own instance cp34_instance(10,'B',2)
  python final/src/run_exact.py --size 100           # scaled instance, M = 100 (also 200, 400)

Appends one row to results/final_exact.csv, writes the exact solution to final/starts/exact_B_<name>.json and the
LP-rounded start (rule: model_b.round_lp) to final/starts/rounded_B_<name>.json.
"""
import argparse, csv, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import milp
import checkpoint4_generator_PUBLIC as gen
from model_b import build_model, solve_lp, round_lp, x_to_assign

COLS = ['instance', 'S', 'M', 'variables', 'rows', 'z_LP', 'lp_fractional_x', 'lp_split_vms', 'rounded_cost',
        'objective', 'checker_cost', 'feasible', 'bound', 'solver_gap', 'proven', 'seconds', 'limit']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--own', action='store_true'); ap.add_argument('--size', type=int)
    ap.add_argument('--limit', type=float, default=120.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'final_exact.csv'))
    a = ap.parse_args()
    if a.own:
        d, name = gen.cp34_instance(10, 'B', 2), 'own'
    else:
        d, name = gen.scaled_instance('B', a.size, 2), str(a.size)
    S, M = d['S'], d['M']
    c, con, bounds, integ = build_model(d)
    zlp, x = solve_lp(d)
    frac = int(((x > 1e-6) & (x < 1 - 1e-6)).sum()); split = int((((x > 1e-6) & (x < 1 - 1e-6)).sum(axis=0) > 0).sum())
    rounded = round_lp(d, x)
    rc, rp = gen.evaluate_B(d, rounded)
    assert not rp, rp
    sd = os.path.join(ROOT, 'final', 'starts'); os.makedirs(sd, exist_ok=True)
    json.dump({'assign': rounded}, open(os.path.join(sd, f'rounded_B_{name}.json'), 'w'))
    # the same call as exact_helper.solve_milp (same options); repeated here only to keep the solution vector
    t0 = time.time()
    res = milp(c, constraints=con, integrality=integ, bounds=bounds,
               options=dict(time_limit=a.limit, mip_rel_gap=1e-6, disp=False))
    sec = time.time() - t0
    obj = float(res.fun) if res.x is not None else None
    bound = getattr(res, 'mip_dual_bound', None)
    gap = max(0.0, (obj - bound) / abs(obj)) if obj is not None and bound is not None else None
    cc, feas = '', ''
    if res.x is not None:
        assign = x_to_assign(d, np.round(res.x))
        cost, problems = gen.evaluate_B(d, assign)
        cc, feas = f'{cost:.9f}', len(problems) == 0
        json.dump({'assign': assign}, open(os.path.join(sd, f'exact_B_{name}.json'), 'w'))
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(COLS)
        w.writerow([name, S, M, len(c), con.A.shape[0], f'{zlp:.9f}', frac, split, f'{rc:.9f}',
                    '' if obj is None else f'{obj:.9f}', cc, feas, '' if bound is None else f'{bound:.9f}',
                    '' if gap is None else f'{gap:.3e}', bool(gap is not None and gap <= 1e-6), f'{sec:.1f}', a.limit])
    print(name, 'zLP', zlp, 'rounded', rc, 'obj', obj, 'bound', bound, 'gap', gap, 'sec', round(sec, 1))


if __name__ == '__main__':
    main()
