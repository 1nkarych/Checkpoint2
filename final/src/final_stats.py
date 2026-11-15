"""Final report, Block 2: statistics of the report, recomputed from per-seed results (team T10, cp34_instance(10, 'B', 2)).

Input  results/extra_scan_all_seeds.json  (written by final/src/extra_scan.py: the final solution of seeds 0-9 of GA and SA,
       re-run with the Checkpoint 4 budget and parameters; field 'cost' = reference cost of the final solution).
Output results/final_stats.json, results/final_table2_check.csv

Computes
  * Table 2 columns for GA and SA (best, mean, worst, std with divisor n-1, gap_ref, gap_LP);
  * the comparison GA - SA: mean difference, paired standard error (difference of seed s of the two methods; the pairing by the
    seed NUMBER is nominal: the two methods do not share a random stream), Welch standard error, paired t-test, Welch t-test,
    Mann-Whitney U, Wilcoxon signed-rank, Levene (centre = median, Brown-Forsythe, and centre = mean);
  * the exact decomposition gap_LP = gap_ref * z_ref / z + 100 (z_ref - z_LP) / z   (the two gaps have different denominators);
  * summary of the neighbourhood scan of all final solutions (transfers, swaps, descent to z_ref);
  * a formula check of the rows of Table 2 that are quoted from the draft (random search, exact solver): final_table2.csv is
    not part of this checkout, so these rows are only checked for internal consistency (gap formulas), not against Checkpoint 4 files.
Usage:  python final/src/final_stats.py
"""
import csv, json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, 'results')
Z_REF = json.load(open(os.path.join(R, 'final_block1_solutions.json')))['z_ref']       # HiGHS, proven
Z_LP = json.load(open(os.path.join(R, 'final_block1_solutions.json')))['z_LP']
# rows quoted from the draft Table 2 (4 decimals as printed): method -> (best, mean, worst, std)
QUOTED = {'Random search': (8.7888, 8.8468, 8.9388, 0.0541), 'Exact solver (HiGHS)': (Z_REF, Z_REF, Z_REF, None)}   # exact row: proven z_ref of the reference solve
PRINTED = {'GA with repair (N = 30)': dict(best=8.3828, mean=8.4143, worst=8.4459, std=0.0184, gap_ref=(0.38, 0.75), gap_lp=(0.89, 1.26)),
           'SA, transfer move': dict(best=8.3919, mean=8.4117, worst=8.4258, std=0.0089, gap_ref=(0.34, 0.51), gap_lp=(0.86, 1.02))}


def gap_ref(z): return 100.0 * (z - Z_REF) / Z_REF
def gap_lp(z): return 100.0 * (z - Z_LP) / z


def col(x):
    x = np.asarray(x, float)
    return dict(best=float(x.min()), mean=float(x.mean()), worst=float(x.max()), std=float(x.std(ddof=1)), range=float(x.max() - x.min()),
                gap_ref_mean=gap_ref(x.mean()), gap_ref_worst=gap_ref(x.max()), gap_LP_mean=gap_lp(x.mean()), gap_LP_worst=gap_lp(x.max()))


def main():
    scan = json.load(open(os.path.join(R, 'extra_scan_all_seeds.json')))
    ga = np.array([r['cost'] for r in scan['GA']]); sa = np.array([r['cost'] for r in scan['SA']])
    assert [r['seed'] for r in scan['GA']] == list(range(10)) == [r['seed'] for r in scan['SA']]
    out = dict(z_ref=Z_REF, z_LP=Z_LP, n_seeds=10, GA=col(ga), SA=col(sa))
    d = ga - sa
    out['comparison_GA_minus_SA'] = dict(
        mean_difference=float(d.mean()), paired_se=float(d.std(ddof=1) / np.sqrt(len(d))),
        welch_se=float(np.sqrt(ga.var(ddof=1) / len(ga) + sa.var(ddof=1) / len(sa))),
        ga_better_seeds=int((ga < sa).sum()), sa_better_seeds=int((sa < ga).sum()), ties=int((ga == sa).sum()),
        paired_t_p=float(stats.ttest_rel(ga, sa).pvalue), welch_t_p=float(stats.ttest_ind(ga, sa, equal_var=False).pvalue),
        mann_whitney_U=float(stats.mannwhitneyu(ga, sa, alternative='two-sided').statistic),
        mann_whitney_p=float(stats.mannwhitneyu(ga, sa, alternative='two-sided').pvalue),
        wilcoxon_signed_rank_p=float(stats.wilcoxon(ga, sa).pvalue),
        levene_median_p=float(stats.levene(ga, sa).pvalue), levene_mean_p=float(stats.levene(ga, sa, center='mean').pvalue),
        std_ratio_SA_over_GA=float(sa.std(ddof=1) / ga.std(ddof=1)),
        note='n = 10 per method; no test is significant at 0.05; Levene p is borderline and is read as a weak indication only')
    dec = {}
    for name, x in (('GA', ga), ('SA', sa)):
        m = x.mean(); integrality = 100 * (Z_REF - Z_LP) / m; method = gap_ref(m) * Z_REF / m
        dec[name] = dict(gap_LP_mean=gap_lp(m), integrality_part=integrality, method_part=method, sum=integrality + method,
                         gap_ref_mean=gap_ref(m))
    dec['integrality_gap_of_exact_optimum_gap_LP'] = gap_lp(Z_REF)         # = 100 (z_ref - z_LP) / z_ref
    out['gap_LP_decomposition'] = dec
    def summ(rows):
        return dict(no_improving_transfer=sum(r['tr_improving'] == 0 for r in rows), at_least_one_improving_transfer=sum(r['tr_improving'] > 0 for r in rows),
                    at_least_one_improving_swap=sum(r['sw_improving'] > 0 for r in rows), no_improving_swap=sum(r['sw_improving'] == 0 for r in rows),
                    descent_transfer_swap_reaches_zref=sum(abs(r['descent_transfer_swap'] - Z_REF) < 1e-9 for r in rows),
                    descent_transfer_only_reaches_zref=sum(abs(r['descent_transfer_only'] - Z_REF) < 1e-9 for r in rows))
    out['neighbourhood_scan'] = dict(GA=summ(scan['GA']), SA=summ(scan['SA']))
    w = {m: max(scan[m], key=lambda r: r['cost']) for m in ('GA', 'SA')}
    out['worst_seed_checks'] = {m: dict(seed=r['seed'], cost=r['cost'], improving_transfers=r['tr_improving'], improving_swaps=r['sw_improving'],
                                        best_swap_delta=r['best_swap_delta'], cost_after_best_swap=r['cost'] + r['best_swap_delta'],
                                        descent_transfer_swap=r['descent_transfer_swap']) for m, r in w.items()}
    json.dump(out, open(os.path.join(R, 'final_stats.json'), 'w'), indent=1)

    rows = []
    for name, x in (('GA with repair (N = 30)', ga), ('SA, transfer move', sa)):
        c = col(x); p = PRINTED[name]
        chk = [round(c['best'], 4) == p['best'], round(c['mean'], 4) == p['mean'], round(c['worst'], 4) == p['worst'], round(c['std'], 4) == p['std'],
               (round(c['gap_ref_mean'], 2), round(c['gap_ref_worst'], 2)) == p['gap_ref'], (round(c['gap_LP_mean'], 2), round(c['gap_LP_worst'], 2)) == p['gap_lp']]
        rows.append([name, f"{c['best']:.6f}", f"{c['mean']:.6f}", f"{c['worst']:.6f}", f"{c['std']:.6f}", f"{c['gap_ref_mean']:.4f}", f"{c['gap_ref_worst']:.4f}",
                     f"{c['gap_LP_mean']:.4f}", f"{c['gap_LP_worst']:.4f}", 'recomputed from per-seed results; matches printed Table 2: ' + str(all(chk))])
    for name, (b, m, wst, sd) in QUOTED.items():
        rows.append([name, b, m, wst, sd if sd is not None else '', f'{gap_ref(m):.4f}', f'{gap_ref(wst):.4f}', f'{gap_lp(m):.4f}', f'{gap_lp(wst):.4f}',
                     ('values QUOTED from the draft (source files not in this checkout); only the gap formulas are recomputed' if name == 'Random search' else 'z_ref of the reference solve (final_block1_solutions.json); gaps recomputed')])
    with open(os.path.join(R, 'final_table2_check.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['method', 'best', 'mean', 'worst', 'std', 'gap_ref_mean', 'gap_ref_worst', 'gap_LP_mean', 'gap_LP_worst', 'provenance'])
        wr.writerows(rows)
    c = out['comparison_GA_minus_SA']
    print(f"GA mean {out['GA']['mean']:.6f} std {out['GA']['std']:.6f} | SA mean {out['SA']['mean']:.6f} std {out['SA']['std']:.6f}")
    print(f"mean diff {c['mean_difference']:.6f}  paired SE {c['paired_se']:.6f}  Welch SE {c['welch_se']:.6f}  GA better on {c['ga_better_seeds']}/10")
    print(f"p: paired t {c['paired_t_p']:.3f}  Welch {c['welch_t_p']:.3f}  Mann-Whitney {c['mann_whitney_p']:.3f}  Wilcoxon {c['wilcoxon_signed_rank_p']:.3f}  "
          f"Levene(median) {c['levene_median_p']:.3f}  Levene(mean) {c['levene_mean_p']:.3f}")
    print('gap_LP decomposition:', json.dumps(dec, indent=1))
    print('scan:', json.dumps(out['neighbourhood_scan']))
    print('worst seeds:', json.dumps(out['worst_seed_checks'], indent=1))
    print('Table 2 check:'); [print(' ', r[0], '|', r[-1]) for r in rows]


if __name__ == '__main__':
    main()
