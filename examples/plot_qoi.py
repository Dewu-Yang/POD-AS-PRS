"""Standalone QoI visualisation script.

Plots the raw QoI time series and its standardised Gaussian-smooth surrogate
for all four cases, reproducing the exact style of the legacy ``plot_qoi.py``
scripts in Example2_NACA4412 / Example2_NACA4412_Cd:

- Two-panel figure  (15 × 4 inches, label_fs = 24)
- Top panel    : raw QoI time series + ±2σ horizontal dashed bounds
- Bottom panel : Gaussian-smoothed, zero-mean, unit-variance signal *q*
                 with ±2σ bounds  (asymmetric lower bound preserved from
                 legacy to match original output exactly)
- LaTeX rendering, Computer-Modern serif font (text.usetex = True)

Output files are written to each case's dedicated ``QoI/`` results folder.

Usage
-----
Run from the ``examples/`` directory::

    python plot_qoi.py
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Global rcParams — matches legacy plot_qoi.py exactly
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "axes.unicode_minus": True,
})

# ---------------------------------------------------------------------------
# Case configurations
# ---------------------------------------------------------------------------
CASES = [
    dict(
        name        = 'Case1_Cylinder_Cd',
        qoi_path    = '../data/Case1_Cylinder/drag_coefficient.dat',
        results_dir = '../results/Case1_Cylinder/QoI',
        qoi_label   = '$C_d$',
        time_label  = r'$t$ /($DU_\infty^{-1}$)',
    ),
    dict(
        name        = 'Case1_Cylinder_Cl',
        qoi_path    = '../data/Case1_Cylinder/lift_coefficient.dat',
        results_dir = '../results/Case1_Cylinder_Cl/QoI',
        qoi_label   = '$C_l$',
        time_label  = r'$t$ /($DU_\infty^{-1}$)',
    ),
    dict(
        name        = 'Case2_NACA4412_Cl',
        qoi_path    = '../data/Case2_NACA4412/lift_coefficient_600-800_truncated.dat',
        results_dir = '../results/Case2_NACA4412/QoI',
        qoi_label   = '$C_l$',
        time_label  = r'$t$ /($CU_\infty^{-1}$)',
    ),
    dict(
        name        = 'Case2_NACA4412_Cd',
        qoi_path    = '../data/Case2_NACA4412/drag_coefficient_600-800_truncated.dat',
        results_dir = '../results/Case2_NACA4412_Cd/QoI',
        qoi_label   = '$C_d$',
        time_label  = r'$t$ /($CU_\infty^{-1}$)',
    ),
]


# ---------------------------------------------------------------------------
# Plot function
# ---------------------------------------------------------------------------
def plot_single_qoi(qoi_path, qoi_label, time_label, results_dir):
    """Produce the two-panel QoI figure for one case.

    Parameters
    ----------
    qoi_path : str
        Two-column (time, value) ``.dat`` file.
    qoi_label : str
        LaTeX label for the raw-QoI y-axis (e.g. ``'$C_d$'``).
    time_label : str
        LaTeX label for the shared time x-axis.
    results_dir : str
        Output directory; created if absent.
    """
    os.makedirs(results_dir, exist_ok=True)

    fc       = np.loadtxt(qoi_path)
    qoi_time = fc[:, 0]
    qoi_vals = fc[:, 1]

    print(f'Mean : {np.mean(qoi_vals):.6f}')
    print(f'Std  : {np.std(qoi_vals):.6f}')
    print(f'N    : {len(qoi_vals)}')

    # Dominant frequency via FFT (dt derived from data)
    dt     = float(qoi_time[1] - qoi_time[0]) if len(qoi_time) > 1 else 1.0
    F_qoi  = np.fft.fft(qoi_vals - np.mean(qoi_vals))
    freqs  = np.fft.fftfreq(len(qoi_vals), d=dt)
    f_peak = freqs[np.argmax(np.abs(F_qoi))]
    print(f'f_peak: {f_peak:.6f}   '
          f'period: {1.0 / f_peak if f_peak != 0 else float("inf"):.4f}')

    scale_smoother = (max(1, int(abs(0.5 / (f_peak * dt))))
                      if f_peak != 0 else 10)

    q = gaussian_filter1d(qoi_vals, sigma=scale_smoother)
    print(f'Mean / std  q: {np.mean(q):.6f} / {np.std(q):.6f}')
    q = (q - np.mean(q)) / np.std(q)
    print(f'Dominant frequency / period: {f_peak}  {f_peak**-1 if f_peak != 0 else "inf"}')

    nsigma   = 2
    label_fs = 24
    start    = 1      # match legacy:  start=1, end=-1
    end      = -1
    t0       = qoi_time[start]
    t1       = qoi_time[end]

    fig = plt.figure(figsize=(15, 4))

    # ── Top panel: raw QoI ──────────────────────────────────────────────────
    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax1.plot(qoi_time, qoi_vals, 'k', linewidth=1)
    mu, sg = np.mean(qoi_vals), np.std(qoi_vals)
    ax1.plot([t0, t1], [mu + nsigma * sg] * 2, 'r--', linewidth=1)
    ax1.plot([t0, t1], [mu - nsigma * sg] * 2, 'r--', linewidth=1)
    ax1.set_xlim([t0, t1])
    ax1.set_ylabel(qoi_label, fontsize=label_fs + 2)
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=label_fs)

    # ── Bottom panel: normalised smooth surrogate ────────────────────────────
    ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1)
    ax2.plot(qoi_time, q, 'k', linewidth=1)
    mq, sq = np.mean(q), np.std(q)
    ax2.plot([t0, t1], [mq + nsigma * sq] * 2,             'r--', linewidth=1)
    ax2.plot([t0, t1], [mq - nsigma * sq, mq + nsigma * sq], 'r--', linewidth=1)
    ax2.set_xlim([t0, t1])
    ax2.tick_params(axis='x', labelsize=label_fs)
    ax2.tick_params(axis='y', labelsize=label_fs)
    ax2.set_ylabel('$q$',       fontsize=label_fs + 2)
    ax2.set_xlabel(time_label,  fontsize=label_fs + 2)

    plt.tight_layout()

    base = os.path.join(results_dir, 'qoi')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'Saved → {base}.[jpg|pdf]')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    for case in CASES:
        print(f'\n── {case["name"]} ' + '─' * (50 - len(case['name'])))
        qoi_path = case['qoi_path']
        if not os.path.exists(qoi_path):
            print(f'  [SKIP] data not found: {qoi_path}')
            continue
        plot_single_qoi(
            qoi_path    = qoi_path,
            qoi_label   = case['qoi_label'],
            time_label  = case['time_label'],
            results_dir = case['results_dir'],
        )
    print('\nDone.')
