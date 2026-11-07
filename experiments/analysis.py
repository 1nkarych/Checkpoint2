"""Analysis of the GA-A runs: traces and diversity (rerun of the 10 final runs: same seed -> same run), worst-seed
checks (Table 2), statistics (Table 1), figures F1 and F2.  Reads results/Telgarin_*.csv, writes results/analysis.json
and report/F1.png, report/F2.png.   Usage: python experiments/analysis.py"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
from src.meta import ga

SURNAME = 'Telgarin'
INST = dict(seed=10, story='B', variant=2)
R = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'report')
os.makedirs(FIG, exist_ok=True)


def read(name):
    with open(os.path.join(R, name)) as fh:
        return list(csv.DictReader(fh))


def stats(costs, zref, zlp):
    c = np.array(costs, float)
    out = dict(best=c.min(), median=float(np.median(c)), mean=c.mean(), worst=c.max(),
               std=c.std(ddof=1) if len(c) > 1 else float('nan'))
    for k in ('best', 'median', 'mean', 'worst'):
        out['gap_ref_' + k] = 100 * (out[k] - zref) / zref
        out['gap_lp_' + k] = 100 * (out[k] - zlp) / out[k]
    return out


def main():
    ref = json.load(open(os.path.join(R, 'ref_values.json')))
    zref, zlp = ref['z_ref'], ref['z_LP']
    d = gen.cp34_instance(INST['seed'], INST['story'], INST['variant'])
    runs = read(f'{SURNAME}_runs.csv'); rnd = read(f'{SURNAME}_random.csv'); nexp = read(f'{SURNAME}_Nexp.csv')
    out = dict(z_ref=zref, z_LP=zlp)

    # ---- Table 1 (feasible runs only enter the statistics; infeasible ones are counted separately)
    feas = [r for r in runs if r['feasible'] == 'True']
    out['table1'] = stats([float(r['cost']) for r in feas], zref, zlp)
    out['table1'].update(n_feasible=len(feas), n_runs=len(runs), mean_seconds=float(np.mean([float(r['seconds']) for r in runs])))
    rf = [r for r in rnd if r['feasible'] == 'True']
    out['random'] = stats([float(r['cost']) for r in rf], zref, zlp)
    out['random'].update(n_feasible=len(rf), n_runs=len(rnd), mean_seconds=float(np.mean([float(r['seconds']) for r in rnd])))
    out['Nexp'] = {}
    for N in (10, 30, 100):
        rows = [r for r in nexp if r['configuration'].endswith(f'N={N}') and r['feasible'] == 'True']
        s = stats([float(r['cost']) for r in rows], zref, zlp); s['n_feasible'] = len(rows)
        s['mean_seconds'] = float(np.mean([float(r['seconds']) for r in rows])); out['Nexp'][N] = s
    # Wilcoxon-free paired comparison GA vs random, same seeds
    ga_by_seed = {int(r['seed']): float(r['cost']) for r in runs}
    rd_by_seed = {int(r['seed']): float(r['cost']) for r in rnd}
    out['ga_better_than_random_seeds'] = sum(ga_by_seed[s] < rd_by_seed[s] for s in ga_by_seed)

    # ---- rerun the final runs (deterministic) for traces and diversity; check against the csv
    traces, distinct, hamming, finals = {}, {}, {}, {}
    for r in runs:
        s = int(r['seed'])
        res = ga.solve(d, INST['variant'], s, int(r['budget']))
        assert abs(res['cost'] - float(r['cost'])) < 1e-9, f'seed {s}: rerun differs from the csv'
        traces[s] = np.array(res['trace'], float); distinct[s] = res['distinct']; hamming[s] = res['hamming']
        finals[s] = res['solution']
    grid = np.arange(100, 100001, 100)
    curves = np.array([np.interp(grid, traces[s][:, 0], traces[s][:, 1]) for s in sorted(traces)])
    mean_c, lo, hi = curves.mean(0), curves.min(0), curves.max(0)
    for thr in (1.0, 0.5, 0.2):
        fin = mean_c[-1]
        idx = int(np.argmax(mean_c <= fin * (1 + thr / 100)))
        out[f'share_budget_within_{thr}pct_of_final'] = float(grid[idx] / 1e5)
    out['mean_cost_at'] = {str(int(e)): float(mean_c[int(e / 100) - 1]) for e in (1000, 5000, 10000, 50000, 90000, 100000)}
    out['mean_gain_last_10pct'] = float(mean_c[int(0.9 * len(grid)) - 1] - mean_c[-1])
    out['seeds_still_improving_in_last_10pct'] = int(sum(curves[i][int(0.9 * len(grid)) - 1] > curves[i][-1] + 1e-12 for i in range(len(curves))))

    # ---- worst seed (feasible run with the largest cost) and the finite list of checks
    wseed = max((int(r['seed']) for r in feas), key=lambda s: ga_by_seed[s])
    sol = finals[wseed]; base, problems = gen.evaluate_B(d, sol); assert not problems
    S, M = d['S'], d['M']
    tr, sw = [], []
    for k in range(M):
        for t in range(S):
            if t == sol[k]: continue
            x = list(sol); x[k] = t; c, pr = gen.evaluate_B(d, x)
            tr.append((c - base, not pr, ('transfer', k, int(sol[k]), t)))
    for i in range(M):
        for j in range(i + 1, M):
            if sol[i] == sol[j]: continue
            x = list(sol); x[i], x[j] = sol[j], sol[i]; c, pr = gen.evaluate_B(d, x)
            sw.append((c - base, not pr, ('swap', i, j, int(sol[i]), int(sol[j]))))
    TOL = 1e-9
    imp_tr = sum(1 for dl, ok, _ in tr if ok and dl < -TOL); imp_sw = sum(1 for dl, ok, _ in sw if ok and dl < -TOL)
    best_tr = min((x for x in tr if x[1]), key=lambda x: x[0]); best_sw = min((x for x in sw if x[1]), key=lambda x: x[0])
    out['worst_seed'] = dict(seed=wseed, cost=base, n_transfers=len(tr), n_swaps=len(sw),
        feasible_transfers=sum(1 for x in tr if x[1]), feasible_swaps=sum(1 for x in sw if x[1]),
        improving_transfers=imp_tr, improving_swaps=imp_sw, tol=TOL,
        best_transfer=dict(delta=best_tr[0], move=best_tr[2]), best_swap=dict(delta=best_sw[0], move=best_sw[2]),
        distinct_first=distinct[wseed][0], distinct_last=distinct[wseed][-1], generations=len(distinct[wseed]),
        hamming_first=hamming[wseed][0], hamming_last=hamming[wseed][-1],
        gap_ref=100 * (base - zref) / zref)
    # where / when the worst seed stopped improving
    tw = traces[wseed]; last_imp = tw[np.argmax(tw[:, 1] <= tw[-1, 1] + 1e-15), 0]
    out['worst_seed']['last_improvement_eval'] = float(last_imp)
    util = np.zeros(S); 
    for k in range(M): util[sol[k]] += d['cpu_req'][k]
    util = util / np.asarray(d['cpu_cap'], float)
    out['worst_seed']['util'] = [float(u) for u in util]
    # across seeds
    out['distinct_first_mean'] = float(np.mean([distinct[s][0] for s in distinct]))
    out['distinct_last_mean'] = float(np.mean([distinct[s][-1] for s in distinct]))
    out['distinct_min_over_all'] = int(min(min(v) for v in distinct.values()))
    hm = np.array([hamming[s] for s in sorted(hamming)]); out['hamming_mean_at'] = {str(g): float(hm[:, g].mean()) for g in (0, 10, 100, 500, 1000, 3000, hm.shape[1] - 1)}
    out['hamming_first'] = float(hm[:, 0].mean()); out['hamming_last'] = float(hm[:, -1].mean())
    dm = np.array([distinct[s] for s in sorted(distinct)])
    json.dump(out, open(os.path.join(R, 'analysis.json'), 'w'), indent=1, default=float)

    # ---- F1
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.fill_between(grid, lo, hi, color='tab:blue', alpha=0.25, label='min-max of 10 runs')
    ax.plot(grid, mean_c, color='tab:blue', lw=1.3, label='mean of 10 runs')
    ax.axhline(zref, color='k', ls='--', lw=0.9, label=f'$z_{{ref}}$ = {zref:.4f}')
    ax.axhline(zlp, color='tab:red', ls=':', lw=1.1, label=f'$z_{{LP}}$ = {zlp:.4f}')
    ax.set_xscale('log'); ax.set_ylim(zlp - 0.03, float(mean_c[int(500/100)]) + 0.1)
    ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost so far'); ax.legend(fontsize=6.5, loc='upper right'); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F1.png'), dpi=220); plt.close(fig)
    # ---- F2: diversity against the generation (N=30), mean of 10 runs and the worst seed
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.0))
    g = np.arange(dm.shape[1])
    ax[0].plot(g, dm.mean(0), color='tab:green', lw=1.2, label='mean of 10 runs'); ax[0].plot(g, dm[sorted(distinct).index(wseed)], color='tab:orange', lw=0.8, label=f'worst seed {wseed}')
    ax[0].set_ylim(0, 31); ax[0].set_xlabel('generation'); ax[0].set_ylabel('distinct individuals'); ax[0].legend(fontsize=6); ax[0].grid(alpha=0.3)
    ax[1].plot(g, hm.mean(0), color='tab:purple', lw=1.2); ax[1].set_xlabel('generation'); ax[1].set_ylabel('mean pairwise Hamming (of 50)'); ax[1].set_xscale('symlog', linthresh=10); ax[1].grid(alpha=0.3)
    for a in ax: a.tick_params(labelsize=6.5); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F2.png'), dpi=220); plt.close(fig)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
