"""Combine matched-seed E0/E2/E3/E4 five-set reports into the required ablation."""
import argparse
import json
from pathlib import Path
import statistics

DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
LABELS = ('UDR-v1', 'U only', 'A only', 'U + A')


def read(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing measured ablation report: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def combine(selection, reports):
    seeds = reports['UDR-v1']['seeds']
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('E4 ablation needs at least three distinct matched seeds.')
    reference = reports['UDR-v1']['runs']['baseline']
    if set(reference[0]) != set(DATASETS):
        raise ValueError('E4 requires exactly the original five datasets.')
    for label, item in reports.items():
        if item['seeds'] != seeds:
            raise ValueError(f'{label} used different Gumbel seeds.')
        for actual, expected in zip(item['runs']['baseline'], reference):
            for dataset in DATASETS:
                for metric in ('psnr', 'ssim'):
                    if abs(actual[dataset][metric] - expected[dataset][metric]) > 1e-8:
                        raise ValueError(f'{label} baseline differs at {dataset}/{metric}.')
    rows = [dict(model='RGB', uncertainty='×', local_alpha='×',
                 depth_confidence='×', average_delta_psnr=0.)]
    kinds = dict(zip(LABELS, (('old', 'global'),
                              (selection['selected_uncertainty']['variant'], 'global'),
                              ('old', selection['selected_alpha']['tag']),
                              (selection['selected_uncertainty']['variant'],
                               selection['selected_alpha']['tag']))))
    per_dataset = {}
    for label in LABELS:
        item = reports[label]
        u, a = kinds[label]
        delta = float(item['average_delta_vs_rgb']['udr'])
        rows.append(dict(model=label, uncertainty=u, local_alpha=a,
                         depth_confidence='✓', average_delta_psnr=delta))
        per_dataset[label] = {
            dataset: dict(delta_psnr=float(item['summary'][dataset]['udr']['delta_psnr_vs_rgb']),
                          psnr_mean=float(item['summary'][dataset]['udr']['psnr']['mean']),
                          psnr_std=float(item['summary'][dataset]['udr']['psnr']['std']),
                          ssim_mean=float(item['summary'][dataset]['udr']['ssim']['mean']),
                          ssim_std=float(item['summary'][dataset]['udr']['ssim']['std']))
            for dataset in DATASETS}
    e4 = rows[-1]['average_delta_psnr']
    urban = per_dataset['U + A']['Urban100']['delta_psnr']
    manga = per_dataset['U + A']['Manga109']['delta_psnr']
    criteria = dict(five_set_gt_003=e4 > .03,
                    five_set_gt_005=e4 > .05,
                    ideal_008_to_010=.08 <= e4 <= .10,
                    urban100_nonnegative=urban >= 0,
                    manga109_positive=manga > 0,
                    minimum_success=e4 > .03 and urban >= 0 and manga > 0)
    return dict(seeds=seeds, ablation=rows, per_dataset=per_dataset,
                e4_criteria=criteria,
                note='All values are read from measured repeated-inference reports; no E4 results are inferred from E2/E3.')


def write_report(path, report):
    path.mkdir(parents=True, exist_ok=True)
    (path/'E4_ablation.json').write_text(json.dumps(report, indent=2,
                                                     ensure_ascii=False, allow_nan=False),
                                           encoding='utf-8')
    lines = ['# E4 UDR-v2 ablation', '',
             f"Matched seeds: {report['seeds']}. Five datasets, Y-channel x4, crop 4.", '',
             '| Model | Uncertainty | Local alpha | Depth confidence | 5-set avg ΔPSNR |',
             '|---|---|---|---|---:|']
    for row in report['ablation']:
        lines.append(f"| {row['model']} | {row['uncertainty']} | {row['local_alpha']} | "
                     f"{row['depth_confidence']} | {row['average_delta_psnr']:+.4f} |")
    lines += ['', '| Dataset | E0 ΔPSNR | E2 ΔPSNR | E3 ΔPSNR | E4 ΔPSNR |',
              '|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        values = [report['per_dataset'][label][dataset]['delta_psnr']
                  for label in LABELS]
        lines.append(f"| {dataset} | " + ' | '.join(f'{v:+.4f}' for v in values) + ' |')
    lines += ['', 'Success criteria: ' + json.dumps(report['e4_criteria'], ensure_ascii=False),
              'Use the JSON for per-dataset PSNR/SSIM mean±std and full reproducibility.', '']
    (path/'E4_ablation.md').write_text('\n'.join(lines), encoding='utf-8')


def self_test():
    def fake(delta):
        runs = {'baseline': [], 'udr': []}
        summary = {}
        for seed in (10, 11, 12):
            runs['baseline'].append({ds: dict(psnr=30 + seed / 100,
                                              ssim=.9 + seed / 10000)
                                     for ds in DATASETS})
            runs['udr'].append({ds: dict(psnr=30 + seed / 100 + delta,
                                         ssim=.9 + seed / 10000)
                                for ds in DATASETS})
        for ds in DATASETS:
            summary[ds] = dict(udr=dict(delta_psnr_vs_rgb=delta,
                                        psnr=dict(mean=statistics.mean(x[ds]['psnr'] for x in runs['udr']), std=.01),
                                        ssim=dict(mean=statistics.mean(x[ds]['ssim'] for x in runs['udr']), std=.001)))
        return dict(seeds=[10, 11, 12], runs=runs, summary=summary,
                    average_delta_vs_rgb=dict(udr=delta))
    selected = dict(selected_uncertainty=dict(variant='U3'),
                    selected_alpha=dict(tag='a1e4'))
    report = combine(selected, dict(zip(LABELS, map(fake, (.02, .04, .03, .06)))))
    assert report['e4_criteria']['minimum_success']
    assert len(report['ablation']) == 5
    bad = dict(zip(LABELS, map(fake, (.02, .04, .03, .06))))
    bad['U only']['runs']['baseline'][0]['Set5']['psnr'] += .1
    try:
        combine(selected, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Unmatched baselines were accepted.')
    print('PASS: five-row E4 ablation and matched-seed/baseline validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='experiments/E4_selection/E4_selection.json')
    parser.add_argument('--e0-summary', default='results/E4_udrv2/E0_five_set/summary.json')
    parser.add_argument('--e2-summary', default='results/E4_udrv2/E2_five_set/summary.json')
    parser.add_argument('--e3-summary', default='results/E4_udrv2/E3_five_set/summary.json')
    parser.add_argument('--e4-summary', default='results/E4_udrv2/E4_five_set/summary.json')
    parser.add_argument('--output', default='results/E4_udrv2/ablation')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    selection = read(args.selection)
    paths = (args.e0_summary, args.e2_summary, args.e3_summary, args.e4_summary)
    reports = {label: read(path) for label, path in zip(LABELS, paths)}
    result = combine(selection, reports)
    write_report(Path(args.output), result)
    print('Saved E4 ablation to', Path(args.output).resolve())


if __name__ == '__main__':
    main()
