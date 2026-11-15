"""Final report, Block 2 (team T10, storyline B, variant 2, instance cp34_instance(10, 'B', 2)).

Table 2 : one row per method of the team (GA with repair, SA with transfer), random search and the exact solver.
Figure F1: best-so-far curves of the team's methods against evaluations, lines at z_LP and z_ref.
Writes   : results/final_table2.{csv,md,tex}, results/final_block2_checks.json, report/final_F1.png, report/final_F1b_boxplot.png

Usage (repository root):
    python final/src/final_block2.py                              # Table 2 from the CSV files of Checkpoint 4 (about 1 s)
    python final/src/final_block2.py --traces --mechanisms        # + rerun of GA and SA (seeds 0-9), F1, worst-seed scans (about 3 min)
Reads    : results/Telgarin_runs.csv (GA, N=30), Telgarin_Nexp.csv, Telgarin_random.csv, Hafizi_runs.csv (SA), abdimazhit_exp.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

R = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'report')
os.makedirs(R, exist_ok=True); os.makedirs(FIG, exist_ok=True)
INST = dict(seed=10, story='B', variant=2)
BUDGET = 100000
SEEDS = list(range(10))


# ----------------------------------------------------------------- reference values (CP3 model, independent rebuild)
def build_model(d):
    """Assignment model of CP3 (storyline B, variant 2): variables y[k,l] (VM k on server l) and t[l] (latency).
    min sum t ; sum_l y[k,l] = 1 ; CPU, memory ; t[l] >= tangent at each of TANGENT_POINTS ; forbidden zone pairs fixed to 0."""
    S, M = d['S'], d['M']
    nY = M * S; n = nY + S
    c = np.zeros(n); c[nY:] = 1.0
    lb = np.zeros(n); ub = np.r_[np.ones(nY), np.full(S, np.inf)]
    for k in range(M):
        if d['vm_required_zone'][k] != -1:
            for l in range(S):
                if d['zone'][l] != d['vm_required_zone'][k]:
                    ub[k * S + l] = 0.0
    A_eq = np.zeros((M, n)); b_eq = np.ones(M)
    for k in range(M):
        A_eq[k, k * S:(k + 1) * S] = 1.0
    rows, rhs = [], []
    for l in range(S):
        r = np.zeros(n); r[l:nY:S] = d['cpu_req']; rows.append(r); rhs.append(d['cpu_cap'][l])
    for l in range(S):
        r = np.zeros(n); r[l:nY:S] = d['mem_req']; rows.append(r); rhs.append(d['mem_cap'][l])
    for l in range(S):
        for a in gen.TANGENT_POINTS:
            s = 1.0 / (1.0 - a) ** 2
            r = np.zeros(n); r[l:nY:S] = s * d['cpu_req'] / d['cpu_cap'][l]; r[nY + l] = -1.0
            rows.append(r); rhs.append(s * a - a / (1.0 - a))
    return dict(c=c, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq, A_ub=np.array(rows), b_ub=np.array(rhs), nY=nY, S=S, M=M)


def reference_values(d, time_limit=120):
    """z_LP (relaxation) and z_ref (exact MILP optimum, HiGHS); also the fractionality of the LP vertex."""
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    m = build_model(d)
    lp = linprog(m['c'], A_ub=m['A_ub'], b_ub=m['b_ub'], A_eq=m['A_eq'], b_eq=m['b_eq'],
                 bounds=list(zip(m['lb'], m['ub'])), method='highs')
    y = lp.x[:m['nY']].reshape(m['M'], m['S'])
    free = m['ub'][:m['nY']] > 0
    frac = (y > 1e-6) & (y < 1 - 1e-6)
    out = dict(z_LP=float(lp.fun), n_free_y=int(free.sum()), n_fractional_y=int(frac.sum()),
               n_vms_split=int(frac.any(axis=1).sum()))
    n = len(m['c'])
    cons = [LinearConstraint(m['A_eq'], m['b_eq'], m['b_eq']), LinearConstraint(m['A_ub'], -np.inf, m['b_ub'])]
    integ = np.r_[np.ones(m['nY']), np.zeros(m['S'])]
    t0 = time.time()
    res = milp(m['c'], constraints=cons, integrality=integ, bounds=Bounds(m['lb'], m['ub']),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-9, disp=False))
    out['z_ref'] = float(res.fun) if res.x is not None else None
    out['z_ref_seconds'] = round(time.time() - t0, 2)
    out['z_ref_proven'] = bool(res.status == 0)
    if res.x is not None:
        out['assign'] = [int(v) for v in np.argmax(res.x[:m['nY']].reshape(m['M'], m['S']), axis=1)]
    return out


# ----------------------------------------------------------------- reading the CP4 files
def read(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        print('MISSING', p); return []
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


def pick(rows, cfg=None):
    return [r for r in rows if cfg is None or r['configuration'] == cfg]


def stats(rows, zref, zlp):
    c = np.array([float(r['cost']) for r in rows])
    sec = np.array([float(r['seconds']) for r in rows])
    feas = sum(str(r['feasible']).strip().lower() == 'true' for r in rows)
    o = dict(n=len(c), best=c.min(), median=float(np.median(c)), mean=c.mean(), worst=c.max(),
             std=c.std(ddof=1) if len(c) > 1 else float('nan'), feasible=f'{feas}/{len(c)}', seconds=sec.mean())
    for k in ('best', 'median', 'mean', 'worst'):
        o['gap_ref_' + k] = 100 * (o[k] - zref) / zref        # as in CP4: relative to z_ref
        o['gap_lp_' + k] = 100 * (o[k] - zlp) / o[k]          # as in CP4: relative to the cost found
    return o


# ----------------------------------------------------------------- neighbourhood scan (CP4, question Q4)
def scan_neighbourhoods(d, assign, tol=1e-9):
    base, pr = gen.evaluate_B(d, list(assign))
    assert not pr, 'solution is infeasible'
    M, S = d['M'], d['S']
    tr = dict(all=0, feas=0, improving=0, best_delta=None, least_bad=None)
    for i in range(M):
        for j in range(S):
            if assign[i] == j: continue
            cand = list(assign); cand[i] = j
            c, p = gen.evaluate_B(d, cand); tr['all'] += 1
            if p: continue
            tr['feas'] += 1; dl = c - base
            if dl < -tol: tr['improving'] += 1
            if dl >= -tol and (tr['least_bad'] is None or dl < tr['least_bad']): tr['least_bad'] = dl
            if tr['best_delta'] is None or dl < tr['best_delta']: tr['best_delta'] = dl
    sw = dict(all=0, feas=0, improving=0, best_delta=None, best_pair=None)
    for i in range(M):
        for j in range(i + 1, M):
            if assign[i] == assign[j]: continue
            cand = list(assign); cand[i], cand[j] = cand[j], cand[i]
            c, p = gen.evaluate_B(d, cand); sw['all'] += 1
            if p: continue
            sw['feas'] += 1; dl = c - base
            if dl < -tol: sw['improving'] += 1
            if sw['best_delta'] is None or dl < sw['best_delta']:
                sw['best_delta'] = dl; sw['best_pair'] = (i, j, int(assign[i]), int(assign[j]))
    return dict(cost=base, transfers=tr, swaps=sw)


def sa_acceptance(result):
    """Acceptance rate of worsening moves in the first and last 10 % of the evaluations (as in sa_i_worst_seed.py)."""
    rec = result['worsening_moves']; win = max(1, (result['evals'] - 1) // 10)
    early = [a for e, _, a in rec if 2 <= e <= win + 1]
    late = [a for e, _, a in rec if result['evals'] - win < e <= result['evals']]
    f = lambda x: (100 * sum(x) / len(x)) if x else None
    return dict(early_pct=f(early), late_pct=f(late), n_early=len(early), n_late=len(late))


# ----------------------------------------------------------------- rerun helpers
def sa_params_from_csv(rows):
    for r in rows:
        if r.get('params'):
            return json.loads(r['params'])
    return None


def run_traces(d, module, params, seeds, label):
    mod = __import__(module, fromlist=['solve'])
    out = []
    for s in seeds:
        t0 = time.time()
        res = mod.solve(d, INST['variant'], s, BUDGET, params=params, start=None)
        cost, pr = check(d, INST['variant'], res['solution'])
        print(f'  {label} seed {s}: {cost:.6f} feasible={not pr} ({time.time() - t0:.1f}s)', flush=True)
        out.append(dict(seed=s, cost=cost, trace=res['trace'], res=res))
    return out


def resample(trace, grid):
    """Best-so-far cost on a common grid of evaluation counts (step function)."""
    ev = np.array([t[0] for t in trace]); cs = np.array([t[1] for t in trace])
    idx = np.searchsorted(ev, grid, side='right') - 1
    return np.where(idx >= 0, cs[np.clip(idx, 0, None)], np.nan)



# ----------------------------------------------------------------- output
def fmt_table(rows, cols, header):
    md = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        md.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', action='store_true', help='rerun GA and SA (seeds 0-9) to draw F1')
    ap.add_argument('--mechanisms', action='store_true', help='rerun worst seeds and scan transfer/swap neighbourhoods')
    ap.add_argument('--skip-exact', action='store_true', help='do not recompute z_LP / z_ref, read results/ref_values.json')
    a = ap.parse_args()
    d = gen.cp34_instance(INST['seed'], INST['story'], INST['variant'])
    print('Instance: S=%d M=%d, CPU %d/%d, memory %d/%d' % (d['S'], d['M'], d['cpu_req'].sum(), d['cpu_cap'].sum(),
                                                          d['mem_req'].sum(), d['mem_cap'].sum()))
    ref_file = json.load(open(os.path.join(R, 'ref_values.json')))
    if a.skip_exact:
        ref = dict(z_LP=ref_file['z_LP'], z_ref=ref_file['z_ref'])
    else:
        ref = reference_values(d)
        print('Recomputed: z_LP = %.6f, z_ref = %.6f (proven=%s, %.1f s); LP vertex: %d/%d fractional y, %d VMs split'
              % (ref['z_LP'], ref['z_ref'], ref['z_ref_proven'], ref['z_ref_seconds'], ref['n_fractional_y'],
                 ref['n_free_y'], ref['n_vms_split']))
        for k in ('z_LP', 'z_ref'):
            if abs(ref[k] - ref_file[k]) > 1e-6:
                print('WARNING: %s differs from results/ref_values.json (%.9f vs %.9f)' % (k, ref[k], ref_file[k]))
    zlp, zref = ref['z_LP'], ref['z_ref']

    # ---- Table 2: GA, SA, random search, exact solver (one row each)
    ga_rows = read('Telgarin_runs.csv'); nexp = read('Telgarin_Nexp.csv'); rnd = read('Telgarin_random.csv')
    sa_rows = read('Hafizi_runs.csv'); rep = read('abdimazhit_exp.csv')
    methods = [('GA with repair (N = 30)', ga_rows), ('SA, transfer move', sa_rows), ('Random search', rnd)]
    table, allstats = [], {}
    for name, rr in methods:
        if not rr: continue
        s = stats(rr, zref, zlp); allstats[name] = s
        table.append(dict(method=name, best='%.4f' % s['best'], mean='%.4f' % s['mean'], worst='%.4f' % s['worst'],
                          std='%.4f' % s['std'], feas=s['feasible'], sec='%.1f' % s['seconds'],
                          gap_ref='%.2f / %.2f' % (s['gap_ref_mean'], s['gap_ref_worst']),
                          gap_lp='%.2f / %.2f' % (s['gap_lp_mean'], s['gap_lp_worst'])))
    table.append(dict(method='Exact solver (HiGHS)', best='%.4f' % zref, mean='%.4f' % zref, worst='%.4f' % zref, std='0',
                      feas='1 run, proven' if ref.get('z_ref_proven') else '1 run, not proven',
                      sec='%.1f' % ref.get('z_ref_seconds', float('nan')) if 'z_ref_seconds' in ref else '-',
                      gap_ref='0.00', gap_lp='%.2f' % (100 * (zref - zlp) / zref)))
    cols = ['method', 'best', 'mean', 'worst', 'std', 'feas', 'sec', 'gap_ref', 'gap_lp']
    header = ['Method', 'best', 'mean', 'worst', 'std', 'feasible runs of 10', 'mean time per run, s',
              'gap_ref % (mean / worst)', 'gap_LP % (mean / worst)']
    with open(os.path.join(R, 'final_table2.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header)
        for r in table: w.writerow([r[c] for c in cols])
    md = fmt_table(table, cols, header)
    open(os.path.join(R, 'final_table2.md'), 'w').write(md + '\n')
    tex = ['\\begin{tabular}{l' + 'r' * (len(cols) - 1) + '}', '\\hline', ' & '.join(header) + ' \\\\', '\\hline']
    for r in table: tex.append(' & '.join(str(r[c]).replace('%', '\\%').replace('_', '\\_') for c in cols) + ' \\\\')
    tex += ['\\hline', '\\end{tabular}']
    open(os.path.join(R, 'final_table2.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\nTable 2 (budget %d, seeds 0-9; std with n-1; gap_ref = 100(z-z_ref)/z_ref, gap_LP = 100(z-z_LP)/z)\n' % BUDGET)
    print(md)

    checks = dict(ref={k: v for k, v in ref.items() if k != 'assign'}, table2=allstats)
    # population-size experiment of GA-A (used in the analysis, not a row of Table 2)
    nrows = {}
    for N in (10, 30, 100):
        rr = pick(nexp, f'GA, repair, N={N}')
        if rr:
            s = stats(rr, zref, zlp); nrows[N] = s
    if nrows:
        checks['N_experiment'] = nrows
        print('\nGA-A population size (same budget): ' + '; '.join('N=%d mean gap_ref %.2f %% std %.4f' % (k, v['gap_ref_mean'], v['std']) for k, v in nrows.items()))
    # repair experiment of GA-P
    if rep and 'share_before' in rep[0]:
        sb = np.array([float(r['share_before']) for r in rep]); sa_ = np.array([float(r['share_after']) for r in rep])
        checks['repair_experiment'] = dict(n_runs=len(rep), offspring_per_run=int(rep[0]['offspring']),
            feasible_share_before_mean=float(sb.mean()), feasible_share_after_mean=float(sa_.mean()),
            repair_failures=int(sum(int(r['repair_failures']) for r in rep)),
            best_cost_before_mean=float(np.mean([float(r['best_cost_before']) for r in rep])),
            best_cost_after_mean=float(np.mean([float(r['best_cost_after']) for r in rep])))
        e = checks['repair_experiment']
        print('Repair experiment (GA-P): feasible offspring %.1f %% -> %.1f %%, failures %d, best of first 1000 offspring %.3f -> %.3f'
              % (100 * e['feasible_share_before_mean'], 100 * e['feasible_share_after_mean'], e['repair_failures'],
                 e['best_cost_before_mean'], e['best_cost_after_mean']))
    # difference between the methods against the spread between seeds (paired by seed)
    if ga_rows and sa_rows:
        g = {int(r['seed']): float(r['cost']) for r in ga_rows}; s_ = {int(r['seed']): float(r['cost']) for r in sa_rows}
        common = sorted(set(g) & set(s_)); diff = np.array([g[k] - s_[k] for k in common])
        gs, ss = allstats['GA with repair (N = 30)'], allstats['SA, transfer move']
        checks['ga_vs_sa'] = dict(seeds=len(common), mean_diff_ga_minus_sa=float(diff.mean()),
                                  se_paired=float(diff.std(ddof=1) / np.sqrt(len(diff))), ga_better_on=int((diff < 0).sum()),
                                  ga_min_max=[gs['best'], gs['worst']], sa_min_max=[ss['best'], ss['worst']],
                                  ga_std=gs['std'], sa_std=ss['std'])
        print('\nGA - SA, paired by seed: mean %.4f, SE %.4f, GA better on %d of %d seeds; GA range %.4f-%.4f, SA range %.4f-%.4f'
              % (diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), (diff < 0).sum(), len(diff), gs['best'], gs['worst'], ss['best'], ss['worst']))

    # ---- F1 and mechanism checks (rerun with the same seeds -> the same runs as in the CSV files)
    curves, runs_cache = {}, {}
    grid = np.unique(np.r_[np.geomspace(1, BUDGET, 400).astype(int), BUDGET])
    if a.traces or a.mechanisms:
        print('\nRerunning GA and SA ...')
        runs_cache['GA'] = run_traces(d, 'src.meta.ga', None, SEEDS, 'GA')
        runs_cache['SA'] = run_traces(d, 'src.meta.sa', sa_params_from_csv(sa_rows), SEEDS, 'SA')
        checks['rerun_matches_csv'] = {}
        for name, rows in (('GA', ga_rows), ('SA', sa_rows)):
            csvc = {int(r['seed']): float(r['cost']) for r in rows}
            bad = [x['seed'] for x in runs_cache[name] if abs(x['cost'] - csvc.get(x['seed'], np.nan)) > 1e-6]
            checks['rerun_matches_csv'][name] = 10 - len(bad)
            print(f'{name}: rerun equals CSV on {10 - len(bad)}/10 seeds' + (f' (differs: {bad})' if bad else ''))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for name, rr in runs_cache.items():
                M_ = np.array([resample(x['trace'], grid) for x in rr])
                curves[name] = dict(mean=np.nanmean(M_, axis=0), lo=np.nanmin(M_, axis=0), hi=np.nanmax(M_, axis=0))
        conv = {}
        for name, c in curves.items():
            fin = c['mean'][-1]
            conv[name] = {f'within_{p}pct_of_final_at_evals': int(grid[np.argmax(c['mean'] <= fin * (1 + p / 100))]) for p in (1.0, 0.5, 0.2)}
        checks['convergence'] = conv
        print('Convergence of the mean curve:', json.dumps(conv))
        checks['sa_start_cost'] = float(runs_cache['SA'][0]['trace'][0][1])
    if a.mechanisms:
        for name in ('GA', 'SA'):
            rr = runs_cache[name]; w = max(rr, key=lambda x: x['cost'])
            sc = scan_neighbourhoods(d, w['res']['solution']); sc['seed'] = w['seed']
            sc['gap_ref_pct'] = 100 * (w['cost'] - zref) / zref
            if sc['swaps']['best_pair']:
                i, j = sc['swaps']['best_pair'][:2]
                sc['swaps']['best_pair_cpu'] = [int(d['cpu_req'][i]), int(d['cpu_req'][j])]
            checks[f'worst_seed_{name}'] = sc
            t, s2 = sc['transfers'], sc['swaps']
            print(f'\n{name}: worst seed {w["seed"]}, cost {w["cost"]:.6f} (gap_ref {sc["gap_ref_pct"]:.2f} %)')
            print(f'  transfers: {t["feas"]}/{t["all"]} feasible, {t["improving"]} improving, least bad delta {t["least_bad"]}')
            print(f'  swaps    : {s2["feas"]}/{s2["all"]} feasible, {s2["improving"]} improving, best delta {s2["best_delta"]} pair {s2["best_pair"]}')
        accs = [sa_acceptance(x['res']) for x in runs_cache['SA']]
        checks['sa_acceptance'] = dict(
            early_mean_pct=float(np.mean([x['early_pct'] for x in accs if x['early_pct'] is not None])),
            late_mean_pct=float(np.mean([x['late_pct'] for x in accs if x['late_pct'] is not None])))
        print('\nSA acceptance of worsening moves: first 10 %% %.1f %%, last 10 %% %.1f %%' % (
            checks['sa_acceptance']['early_mean_pct'], checks['sa_acceptance']['late_mean_pct']))
        gh = [x['res'] for x in runs_cache['GA']]
        checks['ga_diversity_last_generation'] = dict(distinct=[int(r['distinct'][-1]) for r in gh],
                                                      hamming=[float(r['hamming'][-1]) for r in gh])
        print('GA diversity, last generation: distinct', checks['ga_diversity_last_generation']['distinct'],
              'Hamming', [round(x, 1) for x in checks['ga_diversity_last_generation']['hamming']])
    json.dump(checks, open(os.path.join(R, 'final_block2_checks.json'), 'w'), indent=1, default=float)

    # ---- Figure F1: only the methods of the team (curves), z_LP and z_ref as lines
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if curves:
        colors = dict(GA='tab:blue', SA='tab:orange')
        for name, c in curves.items():
            ax.plot(grid, c['mean'], color=colors[name], label=f'{name} (mean of 10 seeds)')
            ax.fill_between(grid, c['lo'], c['hi'], alpha=0.18, color=colors[name], label=f'{name} min-max')
        ax.set_ylim(zlp - 0.05, zlp + 0.9)
    else:
        ax.text(0.5, 0.5, 'run with --traces to draw the curves', ha='center', va='center', transform=ax.transAxes)
    ax.axhline(zref, color='green', ls='--', label='$z_{ref}$ (exact)'); ax.axhline(zlp, color='red', ls=':', label='$z_{LP}$ (relaxation)')
    ax.set_xscale('log'); ax.set_xlabel('evaluations (log scale)'); ax.set_ylabel('best feasible cost so far')
    ax.set_title('F1: best-so-far curves, seeds 0-9'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'final_F1.png'), dpi=170); plt.close(fig)
    # extra figure (not required): final costs per seed
    fig, ax = plt.subplots(figsize=(6, 4.2)); data = [[float(r['cost']) for r in rr] for _, rr in methods if rr]
    ax.boxplot(data, showmeans=True); ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([m for m, rr in methods if rr], fontsize=8)
    for i, v in enumerate(data): ax.scatter(np.random.default_rng(0).normal(i + 1, 0.04, len(v)), v, s=10, color='k', alpha=0.5, zorder=3)
    ax.axhline(zref, color='green', ls='--'); ax.axhline(zlp, color='red', ls=':'); ax.set_ylabel('final cost')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'final_F1b_boxplot.png'), dpi=170); plt.close(fig)
    print('\nWritten: results/final_table2.{csv,md,tex}, results/final_block2_checks.json, report/final_F1.png, report/final_F1b_boxplot.png')


if __name__ == '__main__':
    main()
