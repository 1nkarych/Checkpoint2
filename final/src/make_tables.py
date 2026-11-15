"""Final: every table and figure of the report from the files in results/.
  python final/src/make_tables.py
Reads  results/Telgarin_runs.csv, Telgarin_random.csv, ref_values.json, final_SA_summary.csv, final_trace_*.csv,
       final_GA_repair.csv, final_exact.csv, final_block1.csv
Writes results/final_table2.csv, results/final_table3.csv, report/final_F1.pdf, report/final_F2.pdf and
       report/final_numbers.tex (LaTeX macros with the numbers used in the text)."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, 'results'); REP = os.path.join(ROOT, 'report')
rd = lambda n: list(csv.DictReader(open(os.path.join(R, n))))
ref = json.load(open(os.path.join(R, 'ref_values.json'))); zref, zlp = ref['z_ref'], ref['z_LP']
gr = lambda z: 100 * (z - zref) / zref
gl = lambda z: 100 * (z - zlp) / z

# ---------------- Table 2
def row(name, rows):
    c = np.array([float(r['cost']) for r in rows if r['feasible'] == 'True'])
    return [name, c.min(), c.mean(), c.max(), c.std(ddof=1), f'{len(c)}/{len(rows)}', np.mean([float(r['seconds']) for r in rows]), gr(c.mean()), gl(c.mean())]
t2 = [row('GA, repair ($N=30$)', rd('Telgarin_runs.csv'))]
sa_file = os.path.join(R, 'Hafizi_runs.csv')
if os.path.exists(sa_file):
    t2.append(row('SA, transfer (tuned $T$)', rd('Hafizi_runs.csv')))
else:
    for s in rd('final_SA_summary.csv'):
        if s['configuration'] == 'SA, transfer (tuned $T$)':
            m = float(s['mean'])
            t2.append([s['configuration'], float(s['best']), m, float(s['worst']), float(s['std']), s['feasible'], float(s['seconds']), gr(m), gl(m)])
tun = rd('Shaken_tuning.csv')
t2.append(row('SA, rule-based $T$ (CP4 comparison)', [r for r in tun if r['configuration'] == 'compare-rule']))
t2.append(row('Random search (repaired, GA pair)', rd('Telgarin_random.csv')))
t2.append(row('Random search (no repair, SA pair)', [r for r in tun if r['configuration'] == 'random']))
ex = {r['instance']: r for r in rd('final_exact.csv')}
z = float(ex['own']['objective'])
t2.append(['Exact solver (HiGHS)', z, z, z, 0.0, '1/1', float(ex['own']['seconds']), gr(z), gl(z)])
with open(os.path.join(R, 'final_table2.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['method', 'best', 'mean', 'worst', 'std', 'feasible', 'time_s', 'gap_ref_pct_of_mean', 'gap_LP_pct_of_mean'])
    for r in t2: w.writerow([r[0]] + [f'{v:.6f}' if isinstance(v, float) else v for v in r[1:]])
with open(os.path.join(REP, 'final_table2.tex'), 'w') as fh:
    for r in t2:
        fh.write(f'{r[0]} & {r[1]:.4f} & {r[2]:.4f} & {r[3]:.4f} & {r[4]:.4f} & {r[5]} & {r[6]:.1f} & {abs(r[7]):.2f} & {r[8]:.2f}\\\\\n')

# ---------------- Figure F1
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for name, lab, col in (('GA', 'GA, repair', 'C0'), ('SA', 'SA, transfer only', 'C2'), ('random', 'Random search', 'C1')):
    f = os.path.join(R, f'final_trace_{name}.csv')
    if not os.path.exists(f): continue
    tr = rd(f'final_trace_{name}.csv'); seeds = sorted({int(r['seed']) for r in tr})
    grid = sorted({int(r['evals']) for r in tr})
    grid = [e for e in grid if all(any(int(r['evals']) == e and int(r['seed']) == s for r in tr) for s in seeds[:1])]
    per = {s: dict((int(r['evals']), float(r['best'])) for r in tr if int(r['seed']) == s) for s in seeds}
    grid = [e for e in grid if all(e in per[s] and np.isfinite(per[s][e]) for s in seeds)]
    Y = np.array([[per[s][e] for e in grid] for s in seeds])
    ax.plot(grid, Y.mean(0), color=col, label=f'{lab} (mean of {len(seeds)} seeds, band min-max)')
    ax.fill_between(grid, Y.min(0), Y.max(0), color=col, alpha=0.2, lw=0)
if not os.path.exists(os.path.join(R, 'final_trace_SA.csv')):
    pts = [r for r in rd('final_SA_summary.csv') if r['configuration'] == 'SA curve point']
    ax.plot([int(r['evals']) for r in pts], [float(r['mean']) for r in pts], 's', color='C2', ms=5,
            label='SA, transfer only (mean of 10 seeds; points reported in Checkpoint 4)')
ax.axhline(zref, color='k', ls='--', lw=1, label=f'$z_{{ref}}$ = {zref:.4f}')
ax.axhline(zlp, color='gray', ls=':', lw=1.2, label=f'$z_{{LP}}$ = {zlp:.4f}')
ax.set_xscale('log'); ax.set_xlim(100, 100000); ax.set_ylim(8.3, 9.4)
ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost so far'); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(REP, 'final_F1.pdf')); plt.close(fig)

# ---------------- Table 3
runs = rd('final_GA_repair.csv')
for extra in ('final_SA_scaling.csv',):
    if os.path.exists(os.path.join(R, extra)): runs += rd(extra)
t3 = []
for size in ('100', '200', '400'):
    e = ex[size]
    sub = [r for r in runs if r['size'] == size]
    confs = []
    for r in sub:
        if r['configuration'] not in confs: confs.append(r['configuration'])
    best_known = min([float(e['objective'])] + [float(r['cost']) for r in sub if r['feasible'] == 'True'])
    bk = lambda v: 100 * (v - best_known) / best_known
    zo = float(e['objective'])
    t3.append([size, 'exact (HiGHS)', zo, zo, best_known, bk(zo), bk(zo), 100 * float(e['solver_gap']), e['proven'], float(e['seconds']), '1/1', float(e['z_LP']), float(e['bound'])])
    for cf in confs:
        rr = [r for r in sub if r['configuration'] == cf]
        c = np.array([float(r['cost']) for r in rr if r['feasible'] == 'True'])
        t3.append([size, cf, c.min(), c.mean(), best_known, bk(c.min()), bk(c.mean()), '', '', np.mean([float(r['seconds']) for r in rr]), f'{len(c)}/{len(rr)}', c.std(ddof=1), c.max()])
with open(os.path.join(R, 'final_table3.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['size', 'method', 'best', 'mean', 'best_known', 'gap_best_pct', 'gap_mean_pct', 'solver_gap_pct', 'proven', 'time_s', 'feasible', 'zLP_or_std', 'bound_or_worst'])
    for r in t3: w.writerow([f'{v:.6f}' if isinstance(v, float) else v for v in r])
names = {'GA, repair': 'GA, own start', 'GA, repair, LP-rounded start': 'GA, LP-rounded start',
         'SA, transfer, own start': 'SA, own start', 'SA, transfer, LP-rounded start': 'SA, LP-rounded start'}
label = {'100': 'small', '200': 'medium', '400': 'large'}
with open(os.path.join(REP, 'final_table3.tex'), 'w') as fh:
    for r in t3:
        first = r[1].startswith('exact')
        sz = f"$M={r[0]}$" if first else ''
        if first:
            fh.write(('\\midrule\n' if r[0] != '100' else '') + f"{sz} & exact (HiGHS) & {r[2]:.4f} & -- & {r[4]:.4f} & {r[5]:.2f} & -- & {r[7]:.2f} ({'proven' if r[8] == 'True' else 'not proven'}) & {r[9]:.1f}\\\\\n")
        else:
            fh.write(f"{sz} & {names[r[1]]} & {r[2]:.4f} & {r[3]:.4f} & {r[4]:.4f} & {r[5]:.2f} & {r[6]:.2f} & & {r[9]:.1f}\\\\\n")

# ---------------- Figure F2
fig, ax = plt.subplots(figsize=(5.6, 3.4))
xs = [50, 100, 200, 400]; ys = [float(ex[k]['seconds']) for k in ('own', '100', '200', '400')]
pr = [ex[k]['proven'] == 'True' for k in ('own', '100', '200', '400')]
ax.plot(xs, ys, '-', color='C0', lw=1)
ax.plot([x for x, q in zip(xs, pr) if q], [y for y, q in zip(ys, pr) if q], 'o', color='C0', label='optimum proven (solver gap 0)')
ax.plot([x for x, q in zip(xs, pr) if not q], [y for y, q in zip(ys, pr) if not q], 'x', color='C3', ms=9, mew=2, label='stopped by the limit, not proven')
ax.axhline(120, color='C3', ls='--', lw=1, label='limit 120 s')
for x, y in zip(xs, ys): ax.annotate(f'{y:.1f} s', (x, y), textcoords='offset points', xytext=(4, -12), fontsize=8)
ax.set_xlabel('M (virtual machines); M = 50 is the own instance'); ax.set_ylabel('time of HiGHS, s'); ax.set_xticks(xs)
ax.set_ylim(0, 135); ax.legend(fontsize=8, loc='center right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(REP, 'final_F2.pdf')); plt.close(fig)
