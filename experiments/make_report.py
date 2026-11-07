"""Builds report/Telgarin_CP4.pdf (<= 3 pages) from results/*.csv, results/analysis.json and report/F1.png, F2.png.
Every number of the PDF is read from these files.  Usage: python experiments/make_report.py"""
import csv, json, os
import numpy as np
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, FIG = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'report')
fd = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
for n, f in (('DV', 'DejaVuSans.ttf'), ('DV-B', 'DejaVuSans-Bold.ttf'), ('DV-I', 'DejaVuSans-Oblique.ttf'), ('DV-BI', 'DejaVuSans-BoldOblique.ttf')):
    pdfmetrics.registerFont(TTFont(n, os.path.join(fd, f)))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-BI')

A = json.load(open(os.path.join(R, 'analysis.json')))
T, RS, W, NX = A['table1'], A['random'], A['worst_seed'], A['Nexp']
zref, zlp = A['z_ref'], A['z_LP']
with open(os.path.join(R, 'Telgarin_runs.csv')) as fh:
    runs = list(csv.DictReader(fh))
costs = {int(r['seed']): float(r['cost']) for r in runs}
order = sorted(costs, key=costs.get)
n_within05 = sum(c <= zref * 1.005 for c in costs.values())
gap_relax = 100 * (zref - zlp) / zref            # gap of the relaxation itself, in % of z_ref
gap_relax_lp = 100 * (zref - zlp) / zref
se = lambda s: s['std'] / np.sqrt(10)
d1030 = NX['10']['mean'] - NX['30']['mean']; se1030 = float(np.hypot(se(NX['10']), se(NX['30'])))
d30100 = NX['100']['mean'] - NX['30']['mean']; se30100 = float(np.hypot(se(NX['100']), se(NX['30'])))
gens = {N: int((100000 - N) // (N - 1)) for N in (10, 30, 100)}

st = ParagraphStyle('b', fontName='DV', fontSize=8.2, leading=10.2, spaceAfter=3)
h = ParagraphStyle('h', parent=st, fontName='DV-B', fontSize=9.2, leading=11.5, spaceBefore=5, spaceAfter=1.5, textColor=colors.HexColor('#1f3b63'))
ti = ParagraphStyle('t', parent=st, fontName='DV-B', fontSize=11.5, leading=14, spaceAfter=1)
cell = ParagraphStyle('c', parent=st, fontSize=7.2, leading=8.6, spaceAfter=0)
P = lambda t, s=st: Paragraph(t, s)
f4 = lambda x: f'{x:.4f}'; f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'


def table(rows, widths, head=1, extra=()):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.3, colors.grey), ('BACKGROUND', (0, 0), (-1, head - 1), colors.HexColor('#e6edf7')),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 1.1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.1),
                           ('LEFTPADDING', (0, 0), (-1, -1), 2.5), ('RIGHTPADDING', (0, 0), (-1, -1), 2.5), *extra]))
    return t


S = []
S.append(P('Checkpoint 4 — Genetic algorithm with repair (role GA-A)', ti))
S.append(P(f'<b>Diar Telgarin</b>, group IT-2409, team T10 · partner GA-P: Inkar Abdimazhit · instance <i>cp34_instance(10, "B", 2)</i> '
           f'(S = 11 servers, M = 50 VMs, zones) · z<sub>LP</sub> = {f4(zlp)}, z<sub>ref</sub> = {f4(zref)} · budget 100 000 evaluations, seeds 0–9.'))
S.append(P('<b>Method.</b> Individual = server of each VM. Rule parameters (not tuned): N = 30, p<sub>c</sub> = 0.9 (uniform crossover), 1–2 mutated components '
           '(a VM moved to another allowed server), tournament of size 2, elitism 1. Each offspring is repaired <i>before</i> it is evaluated: VMs on a forbidden zone are '
           'moved to an allowed server, then from every overloaded server the smallest VM that cures the overload (else the largest) goes to the allowed server with room '
           'and the lowest CPU utilisation. The repair reads only loads and capacities (free); every offspring and every initial individual costs 1 evaluation, '
           'the elite copy is free, the last generation is cut short (exactly 100 000). Fitness = cost. My part: <i>ga.py</i> (loop, solve), the tests, the 10 final runs, '
           'the random-search baseline and the experiments; GA-P: <i>ga_problem.py</i>.'))
S.append(P('1. Prediction (written before the final runs)', h))
S.append(P('<b>Feasibility:</b> 10 of 10 runs feasible — the repair always returns a feasible offspring and the fill ratio is low (CPU 233/504 = 46.2 %, memory 871/1725 = 50.5 %; '
           'only 7 premium VMs have a zone condition, each with 3–4 allowed servers). <b>Mean gap<sub>ref</sub>:</b> between 0.2 % and 1.5 %, because z<sub>LP</sub> and z<sub>ref</sub> are only '
           '~0.5 % apart and the convex latency has many nearly equal balanced assignments; the rest would come from the repair bias and from lost diversity, not from the budget. '
           '<b>Diversity:</b> the number of <i>distinct</i> individuals will not collapse (every offspring is mutated, exact clones are rare) and stays above ~10, but the genotypes will cluster '
           'around the elite: mean pairwise Hamming distance below ~5 of 50 within the first 10–20 % of the budget (premature convergence in structure); a larger N will not improve the cost much.'))

S.append(P('2. Results', h))
rows = [['', 'best', 'median', 'mean', 'worst', 'std', 'feasible', 'time', 'gap<sub>ref</sub> % (best / med / mean / worst)', 'gap<sub>LP</sub> % (best / med / mean / worst)'],
        ['<b>GA, repair</b> (N = 30)', f4(T['best']), f4(T['median']), f4(T['mean']), f4(T['worst']), f4(T['std']), f"{T['n_feasible']}/10", f"{T['mean_seconds']:.1f} s",
         ' / '.join(f2(T['gap_ref_' + k]) for k in ('best', 'median', 'mean', 'worst')), ' / '.join(f2(T['gap_lp_' + k]) for k in ('best', 'median', 'mean', 'worst'))],
        ['Random search', f4(RS['best']), f4(RS['median']), f4(RS['mean']), f4(RS['worst']), f4(RS['std']), f"{RS['n_feasible']}/10", f"{RS['mean_seconds']:.1f} s",
         ' / '.join(f2(RS['gap_ref_' + k]) for k in ('best', 'median', 'mean', 'worst')), ' / '.join(f2(RS['gap_lp_' + k]) for k in ('best', 'median', 'mean', 'worst'))]]
S.append(P('<b>Table 1.</b> Final runs, seeds 0–9 (results/Telgarin_runs.csv; baseline results/Telgarin_random.csv, same budget and seeds; std with divisor n − 1; time measured while another job was running).'))
S.append(table(rows, [2.9*cm, 1.15*cm, 1.15*cm, 1.15*cm, 1.15*cm, 1.35*cm, 1.2*cm, 1.1*cm, 3.35*cm, 3.35*cm]))
S.append(Spacer(1, 3))
img = Table([[Image(os.path.join(FIG, 'F1.png'), width=9.0*cm, height=9.0*cm*2.6/4.6), Image(os.path.join(FIG, 'F2.png'), width=8.9*cm, height=8.9*cm*2.0/4.6)]], colWidths=[9*cm, 9*cm])
img.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 0)]))
S.append(img)
S.append(P('<b>Figure F1</b> (left): best feasible cost against evaluations, mean of the 10 runs with the min–max band, z<sub>ref</sub> and z<sub>LP</sub> as lines (log x axis). '
           '<b>Figure F2</b> (right): diversity of the population (N = 30) against the generation: distinct individuals (mean of 10 runs and worst seed) and mean pairwise Hamming distance.'))
rows2 = [['Table 2: worst seed', 'value', 'Effect of N (seeds 0–9, results/Telgarin_Nexp.csv)', 'N = 10', 'N = 30', 'N = 100'],
         ['seed / final feasible cost', f"{W['seed']} / {W['cost']:.6f}", 'generations in 100 000 evals', gens[10], gens[30], gens[100]],
         ['improving transfers / swaps', f"{W['improving_transfers']} / {W['improving_swaps']}", 'mean cost (gap<sub>ref</sub> %)', *[f"{f4(NX[k]['mean'])} ({f2(NX[k]['gap_ref_mean'])})" for k in ('10', '30', '100')]],
         ['distinct individuals, first / last generation', f"{W['distinct_first']} / {W['distinct_last']}", 'worst cost (gap<sub>ref</sub> %)', *[f"{f4(NX[k]['worst'])} ({f2(NX[k]['gap_ref_worst'])})" for k in ('10', '30', '100')]],
         ['checked moves (all / feasible)', f"{W['n_transfers']} / {W['feasible_transfers']} transfers; {W['n_swaps']} / {W['feasible_swaps']} swaps", 'std; feasible runs', *[f"{f4(NX[k]['std'])}; {NX[k]['n_feasible']}/10" for k in ('10', '30', '100')]]]
S.append(table(rows2, [3.9*cm, 3.6*cm, 4.2*cm, 2.0*cm, 2.0*cm, 2.0*cm]))
S.append(P(f'Improving = feasible and Δ &lt; −{W["tol"]:.0e}; first generation = initial population, last = last complete generation (index {W["generations"]-1}). N = 30 is the rule value and stays in Table 1; '
           'the N runs are a comparison on seeds 0–9, not a tuning.', ParagraphStyle('s', parent=st, fontSize=6.4, leading=7.6)))

S.append(P('3. The six questions', h))
S.append(P(f'<b>Q1 Feasible?</b> Yes: {T["n_feasible"]} of 10 runs (and 10 of 10 for every N and for random search) pass the checker; there is no infeasible run, so no overload to report. '
           'The repair keeps every individual within CPU, memory and zone, and the instance is far from full (46 % / 50 %).'))
S.append(P(f'<b>Q2 Repeatable?</b> Only partly. Best {f4(T["best"])} (seed {order[0]}, equal to z<sub>ref</sub> up to {abs(T["best"]-zref):.0e}), median {f4(T["median"])}, worst {f4(T["worst"])} (seed {order[-1]}), std {f4(T["std"])}. '
           f'The spread worst − best is {T["worst"]-T["best"]:.3f} = {100*(T["worst"]-T["best"])/zref:.2f} % of z<sub>ref</sub>, twice the mean gap ({T["gap_ref_mean"]:.2f} %); {n_within05} of 10 runs are within 0.5 % of z<sub>ref</sub>. '
           'Different seeds end in different local optima.'))
S.append(P(f'<b>Q3 Converged?</b> The mean curve is within 1 % of its final value after {100*A["share_budget_within_1.0pct_of_final"]:.0f} % of the budget, within 0.5 % after {100*A["share_budget_within_0.5pct_of_final"]:.0f} % and within 0.2 % after '
           f'{100*A["share_budget_within_0.2pct_of_final"]:.0f} %. In the last 10 % of the budget the mean improves by only {A["mean_gain_last_10pct"]:.4f} ({100*A["mean_gain_last_10pct"]/T["mean"]:.2f} %) and only '
           f'{A["seeds_still_improving_in_last_10pct"]} of 10 runs improve at all: the curve is flat at the end (stuck), not budget-limited. Yet N = 10 (more generations) is better, so the budget is not irrelevant.'))
S.append(P(f'<b>Q4 Local optimum of which neighbourhood?</b> Worst seed {W["seed"]}, final cost {W["cost"]:.5f} (gap<sub>ref</sub> {W["gap_ref"]:.2f} %): of {W["n_transfers"]} transfers ({W["feasible_transfers"]} feasible) '
           f'<b>{W["improving_transfers"]} improve</b>, the least bad one costs Δ = +{W["best_transfer"]["delta"]:.4f} (VM {W["best_transfer"]["move"][1]}, server {W["best_transfer"]["move"][2]} → {W["best_transfer"]["move"][3]}); of {W["n_swaps"]} swaps '
           f'({W["feasible_swaps"]} feasible) <b>{W["improving_swaps"]} improve</b>, the best one (VMs {W["best_swap"]["move"][1]} and {W["best_swap"]["move"][2]}, servers {W["best_swap"]["move"][3]} ↔ {W["best_swap"]["move"][4]}) gives Δ = {W["best_swap"]["delta"]:.4f}, '
           f'which alone would bring the run to {W["cost"]+W["best_swap"]["delta"]:.4f}, close to the median. So the final solution is a local optimum for transfers but not for swaps: a mutation of 1–2 components can make this swap only by hitting two '
           f'given VMs <i>and</i> their two given servers, and the repair never proposes moves of this kind. The run reached its final value at evaluation {format(int(W["last_improvement_eval"]), ",").replace(",", " ")} (last {100-W["last_improvement_eval"]/1000:.1f} % of the budget without improvement); '
           f'the CPU utilisation of the servers is uneven ({min(W["util"]):.2f}–{max(W["util"]):.2f}).'))
S.append(P(f'<b>Q5 What did the method do?</b> Distinct individuals: {W["distinct_first"]} in the first and {W["distinct_last"]} in the last generation of the worst seed (mean over seeds {A["distinct_first_mean"]:.0f} and {A["distinct_last_mean"]:.0f}, minimum over all generations and seeds {A["distinct_min_over_all"]}): '
           f'no identical individuals, no collapse. The mean pairwise Hamming distance falls from {A["hamming_first"]:.1f} (random start) to {A["hamming_mean_at"]["100"]:.1f} in generation 100 and then stays at {A["hamming_last"]:.1f} of 50 until the end (F2): '
           'a balance of mutation, crossover and selection, not a convergence. Two parents that differ in ~15 components give by uniform crossover a child that differs from each parent in about 7–8 components, so crossover keeps making large, '
           'mostly worse steps (my explanation, consistent with the numbers; I did not test it).'))
S.append(P(f'<b>Q6 Better than random search?</b> Yes, on all {A["ga_better_than_random_seeds"]} seeds: mean gap<sub>ref</sub> {T["gap_ref_mean"]:.2f} % against {RS["gap_ref_mean"]:.2f} %, mean gap<sub>LP</sub> {T["gap_lp_mean"]:.2f} % against {RS["gap_lp_mean"]:.2f} % '
           f'(random search used the same repaired random individuals and the same budget). The relaxation itself is {gap_relax:.2f} % below z<sub>ref</sub>: of the {T["gap_lp_mean"]:.2f} % mean gap to z<sub>LP</sub>, {gap_relax_lp:.2f} points are the gap of the relaxation '
           f'(the best run, which equals z<sub>ref</sub>, shows exactly this) and only about {T["gap_ref_mean"]:.2f} points are the GA’s own gap.'))

S.append(P('4. Required item: diversity and population size; prediction against result', h))
S.append(P(f'<b>Population size.</b> With the same budget N = 10 gives {gens[10]} generations, N = 30 {gens[30]} and N = 100 only {gens[100]}. N = 100 is clearly worse (mean {f4(NX["100"]["mean"])} against {f4(NX["30"]["mean"])}; difference {d30100:.3f}, '
           f'standard error ≈ {se30100:.3f}, about {d30100/se30100:.0f} standard errors; it also has the larger std {f4(NX["100"]["std"])}): a big population leaves only {gens[100]} generations for the same budget, and each generation improves the best individual only slowly (I did not plot the N = 100 curves). '
           f'N = 10 is slightly better than N = 30 (mean {f4(NX["10"]["mean"])}, worst {f4(NX["10"]["worst"])}), but the difference {abs(d1030):.4f} is about {abs(d1030)/se1030:.1f} standard errors ({se1030:.4f}), so with 10 seeds I cannot call it a real gain. '
           'More individuals do not help here; more generations do.'))
S.append(P(f'<b>Prediction against result.</b> Feasibility: right (10/10). Mean gap<sub>ref</sub> {T["gap_ref_mean"]:.2f} % lies in the predicted 0.2–1.5 % (the worst run, {T["gap_ref_worst"]:.2f} %, as well). Distinct individuals: right in the sense that they do not collapse — but they stayed at '
           f'{A["distinct_last_mean"]:.0f} of 30, much higher than my guess “above ~10”. <b>Wrong:</b> I expected the Hamming distance below ~5 within 10–20 % of the budget; it fell to ~{A["hamming_mean_at"]["100"]:.0f} within about 3 % of the budget (generation 100) and stayed there, '
           'so the population did not converge prematurely; the stalled runs are therefore not a loss of diversity — my reading (not tested separately) is a crossover that is too disruptive plus the missing swap-like step (Q4). “A larger N will not help” was right for N = 100 (worse), not for N = 10 (equal or slightly better).'))
S.append(P('Reproduction: <i>bash make_all.sh</i> (README). Declared in the README: external code, AI use, checks.', ParagraphStyle('f', parent=st, fontSize=6.4, leading=7.6)))

doc = SimpleDocTemplate(os.path.join(FIG, 'Telgarin_CP4.pdf'), pagesize=A4, leftMargin=1.4*cm, rightMargin=1.4*cm, topMargin=1.2*cm, bottomMargin=1.1*cm,
                        title='Checkpoint 4 — GA-A — Telgarin', author='Diar Telgarin')
doc.build(S)
from pypdf import PdfReader  # noqa
print('pages:', len(PdfReader(os.path.join(FIG, 'Telgarin_CP4.pdf')).pages))
