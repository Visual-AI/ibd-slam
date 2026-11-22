"""Collect slam_result.json / recon_result_d8.json of all runs under a directory into a markdown table."""
import glob
import json
import os
import sys
import warnings

import numpy as np

COLS = [('ATE', 'ate_cm', 'slam'), ('Depth L1', 'depth_l1_cm', 'slam'), ('PSNR', 'psnr', 'slam'),
        ('SSIM', 'ssim', 'slam'), ('Acc', 'acc_cm', 'rec'), ('Comp', 'comp_cm', 'rec'),
        ('Comp. Ratio', 'comp_ratio', 'rec'), ('Acc (vis)', 'acc_cm_vis', 'rec'),
        ('Comp (vis)', 'comp_cm_vis', 'rec'), ('Comp. Ratio (vis)', 'comp_ratio_vis', 'rec')]


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else {}


def fmt(row):
    return ' | '.join('-' if v is None or np.isnan(v) else (f'{v:.3f}' if k == 'ssim' else f'{v:.2f}')
                      for v, (_, k, _) in zip(row, COLS))


def main(root):
    rows = {}
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        r = {'slam': load(os.path.join(d, 'slam_result.json')), 'rec': load(os.path.join(d, 'recon_result_d8.json'))}
        if r['slam']:
            rows[os.path.basename(d)] = [r[src].get(k) if r[src].get(k) is not None else np.nan for _, k, src in COLS]
    print('| sequence | ' + ' | '.join(c[0] for c in COLS) + ' |')
    print('|---' * (len(COLS) + 1) + '|')
    for name, row in rows.items():
        print(f'| {name} | {fmt(row)} |')
    if len(rows) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(np.array(list(rows.values()), float), 0)
        print(f'| **mean** | {fmt(mean)} |')


if __name__ == '__main__':
    main(sys.argv[1])
