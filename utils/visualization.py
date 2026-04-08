"""
High-level visualisation helpers for the POD-AS-PRS workflow.

Functions
---------
plot_eigenvectors_weighted     Weighted combination of the leading eigenvectors.
plot_pod_importance            Horizontal bar chart of POD mode importance scores.
plot_polynomial_cv             Cross-validation R² and RMSE vs polynomial order.
plot_response_surface_2d       3-D surface + 2-D contour of the response surface.
validate_response_surface      True-vs-predicted and residual plots for the PRS.
compare_rom_fom_predictions    ROM vs FOM time-series and scatter comparison.
plot_subspace_polynomial_heatmap  R² heatmap over subspace dimension × polynomial order.
plot_interaction_heatmap       Lower-triangle modal interaction heatmap.
"""

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.colors as mcolors

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.titlesize": 26,
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
mpl.rcParams['axes.unicode_minus'] = False


def plot_eigenvectors_weighted(eigenvecs, eigenvals, num_pod_coeffs,
                                save_dir='./results/eigenvectors', top_n=None):
    """Plot the eigenvalue-weighted combination of the two leading eigenvectors.

    Parameters
    ----------
    eigenvecs : ndarray
        m-by-k matrix of eigenvectors (columns).
    eigenvals : ndarray
        k-by-1 vector of corresponding eigenvalues.
    num_pod_coeffs : int
        Number of POD modes used (for labelling the output file).
    save_dir : str
        Directory for saving the figure.
    top_n : int, optional
        If given, display only the first ``top_n`` components.

    Returns
    -------
    ndarray
        The (possibly truncated) weighted eigenvector.
    """
    from lib.active_subspaces.utils.plotters import plot_opts, show_plot

    os.makedirs(save_dir, exist_ok=True)

    W = eigenvecs
    weighted_eigenvec = np.zeros(W.shape[0])
    for i in range(2):
        weighted_eigenvec += W[:, i] * eigenvals[i]
        print(f"Eigenvector {i+1}: {W[:, i]}")
        print(f"Weight (eigenvalue {i+1}): {eigenvals[i]}")
    print(f"Weighted eigenvector: {weighted_eigenvec}")

    norm = np.linalg.norm(weighted_eigenvec)
    if norm > 0:
        weighted_eigenvec = weighted_eigenvec / norm

    if top_n is not None:
        weighted_eigenvec_out = weighted_eigenvec[:top_n]
        indices = np.arange(1, top_n + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(indices, weighted_eigenvec_out, 'ko-', markersize=12)
        plt.xlabel('Modes', fontsize=24)
        plt.ylabel('Weighted Eigenvector Components', fontsize=24)
        plt.title(
            f'$C_d$ \\_\\{{ {num_pod_coeffs}\\}}, Combined evec 1+2 (Top {top_n})',
            fontsize=24,
        )
        plt.xticks(indices, fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True)
        plt.axis([1, top_n, -1, 1])
        plt.tight_layout()
        figname = os.path.join(
            save_dir, f'weighted_eigenvectors_{num_pod_coeffs}_top{top_n}.jpg'
        )
    else:
        weighted_eigenvec_out = weighted_eigenvec
        m = weighted_eigenvec.shape[0]
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, m + 1), weighted_eigenvec, 'ko-', markersize=12)
        plt.xlabel('Index', fontsize=24)
        plt.ylabel('Weighted Eigenvector Components', fontsize=24)
        plt.title(
            f'$C_d$ \\_\\{{ {num_pod_coeffs}\\}}, Combined evec 1+2', fontsize=24
        )
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True)
        plt.axis([1, m, -1, 1])
        plt.tight_layout()
        figname = os.path.join(
            save_dir, f'weighted_eigenvectors_{num_pod_coeffs}.jpg'
        )

    plt.savefig(figname, dpi=650, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(figname.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight',
                pad_inches=0.0)
    plt.close()
    return weighted_eigenvec_out


def plot_pod_importance(num_pod_coeffs, pod_importance,
                        save_dir='./results/Importance', top_n=None):
    """Horizontal bar chart of POD mode importance scores ranked by contribution.

    Parameters
    ----------
    num_pod_coeffs : int
        Total number of POD modes considered.
    pod_importance : ndarray
        Un-sorted array of importance scores, one per POD mode.
    save_dir : str
        Directory for saving the figure.
    top_n : int, optional
        If given, display only the ``top_n`` most important modes.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    sorted_indices = np.argsort(pod_importance)[::-1]
    sorted_importance = pod_importance[sorted_indices]

    if top_n is not None and top_n < num_pod_coeffs:
        display_n = top_n
        sorted_indices = sorted_indices[:top_n]
        sorted_importance = sorted_importance[:top_n]
    else:
        display_n = num_pod_coeffs

    fig, ax = plt.subplots(figsize=(6, 8))

    inverted_importance = sorted_importance[::-1]
    mode_indices_inv = [sorted_indices[i] for i in reversed(range(len(sorted_indices)))]

    if np.isnan(sorted_importance).any() or np.isinf(sorted_importance).any():
        xmax = 1.0
        print("Warning: NaN or Inf detected in importance scores; using default x-range.")
    else:
        xmax = max(sorted_importance) * 1.2
    ax.set_xlim(0, xmax)

    bars = ax.barh(np.arange(len(sorted_indices)), inverted_importance,
                   height=0.6, color='lightgray', edgecolor='black', hatch='/////')

    for bar, val in zip(bars, inverted_importance):
        ax.text(val + xmax * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=20, color='black')

    mode_labels = [f'Mode {mode_indices_inv[i]+1}' for i in range(len(sorted_indices))]
    ax.set_yticks(np.arange(len(sorted_indices)))
    ax.set_yticklabels(mode_labels, fontsize=24)
    ax.set_xlabel('Relative importance', fontsize=26)
    ax.grid(False, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    fig.tight_layout()

    if top_n is not None and top_n < num_pod_coeffs:
        fname = f'pod_mode_importance_{num_pod_coeffs}_top{top_n}.jpg'
    else:
        fname = f'pod_mode_importance_{num_pod_coeffs}.jpg'

    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=650, bbox_inches='tight')
    plt.savefig(out_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()
    return out_path


def plot_polynomial_cv(n_values, r2_values, rmse_values, best_n,
                        best_score, best_rmse,
                        save_dir='./results/Polynomial_CV'):
    """Two-panel plot of polynomial order cross-validation results.

    Parameters
    ----------
    n_values : list of int
        Polynomial orders tested.
    r2_values, rmse_values : list of float
        Corresponding test-set R² and RMSE values.
    best_n : int
        Polynomial order with the best validation score.
    best_score, best_rmse : float
        Best R² and RMSE.
    save_dir : str
        Directory for saving the figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(n_values, r2_values, 'o-', color='blue', linewidth=2)
    ax1.set_xlabel('Polynomial Order N')
    ax1.set_ylabel('Test Set R²')
    ax1.set_title('R² Variation with Polynomial Order')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.plot(best_n, best_score, 'r*', markersize=10)
    ax1.annotate(
        f'Best: N={best_n}\nR²={best_score:.6f}',
        xy=(best_n, best_score),
        xytext=(best_n - 1, best_score - 0.001),
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    ax1.set_xticks(n_values)

    ax2.plot(n_values, rmse_values, 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Polynomial Order N')
    ax2.set_ylabel('Test Set RMSE')
    ax2.set_title('RMSE Variation with Polynomial Order')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.plot(best_n, best_rmse, 'r*', markersize=10)
    ax2.annotate(
        f'Best: N={best_n}\nRMSE={best_rmse:.8f}',
        xy=(best_n, best_rmse),
        xytext=(best_n - 1, best_rmse + 0.00002),
        arrowprops=dict(arrowstyle='->', color='red'),
    )
    ax2.set_xticks(n_values)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'polynomial_cv_results.jpg')
    plt.savefig(out_path, dpi=650, bbox_inches='tight')
    plt.savefig(out_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()
    return out_path


def plot_response_surface_2d(xx, yy, zz, y, f, results_dir='./results'):
    """3-D surface and 2-D contour plots of a 2-D polynomial response surface.

    Parameters
    ----------
    xx, yy : ndarray
        2-D meshgrid arrays for the two active variables.
    zz : ndarray
        Response surface values on the meshgrid.
    y : ndarray, shape (N, 2)
        Active-variable coordinates of the training samples.
    f : ndarray, shape (N,)
        Corresponding QoI values.
    results_dir : str
        Directory for saving the figures.

    Returns
    -------
    str, str
        Paths to the 3-D surface figure and the 2-D contour figure.
    """
    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax.scatter(y[:, 0], y[:, 1], f, c=f, cmap='viridis',
               edgecolor='k', s=50, alpha=0.6)
    ax.set_xlabel('Active variable 1', fontsize=26, labelpad=15)
    ax.set_ylabel('Active variable 2', fontsize=26, labelpad=15)
    ax.zaxis.set_tick_params(pad=10)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15,
                        pad=0.1, location='right')
    cbar.ax.tick_params(labelsize=24)
    ax.view_init(elev=20, azim=45)

    surf3d_path = os.path.join(results_dir, 'response_surface_2d.jpg')
    plt.savefig(surf3d_path, dpi=650, bbox_inches='tight')
    plt.savefig(surf3d_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, cmap='viridis', levels=15)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=24)
    plt.scatter(y[:, 0], y[:, 1], c=f.flatten(), cmap='viridis',
                edgecolor='k', s=80, alpha=0.8)
    plt.xlabel('Active variable 1', fontsize=26)
    plt.ylabel('Active variable 2', fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    contour_path = os.path.join(results_dir, 'response_surface_contour.jpg')
    plt.savefig(contour_path, dpi=650, bbox_inches='tight')
    plt.savefig(contour_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()

    return surf3d_path, contour_path


def validate_response_surface(model, X_test, y_test_true,
                               save_dir='./results/RS_Validation'):
    """Evaluate a polynomial response-surface model on a held-out test set.

    Produces a true-vs-predicted scatter plot, a residual plot, and (when the
    input has two columns) a 2-D spatial residual map.

    Parameters
    ----------
    model : object
        Response surface model with a ``predict(X)`` method returning
        ``(y_pred, …)``.
    X_test : ndarray, shape (N, d)
        Test-set active-variable coordinates.
    y_test_true : ndarray, shape (N,)
        Ground-truth QoI values.
    save_dir : str
        Directory for saving figures and the metrics text file.

    Returns
    -------
    dict
        Dictionary with keys ``'r2'``, ``'rmse'``, ``'mae'``.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_test_pred = model.predict(X_test)[0]
    r2   = r2_score(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae  = np.mean(np.abs(y_test_true - y_test_pred))

    metrics_path = os.path.join(save_dir, 'validation_metrics.txt')
    with open(metrics_path, 'w') as fp:
        fp.write(f"{'Model':<16} | {'R2':<12} | {'RMSE':<12} | {'MAE':<12}\n")
        fp.write('-' * 60 + '\n')
        fp.write(f"{'Polynomial':<16} | {r2:<12.6f} | {rmse:<12.8f} | {mae:<12.8f}\n")

    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    pad = (max_val - min_val) * 0.05

    plt.figure(figsize=(9, 9))
    plt.scatter(y_test_true, y_test_pred, alpha=0.7, edgecolor='k')
    plt.plot([min_val - pad, max_val + pad],
             [min_val - pad, max_val + pad], 'k--', lw=2)
    plt.xlabel('True Values', fontsize=24)
    plt.ylabel('Predicted Values', fontsize=24)
    plt.title('Polynomial Response Surface: Validation', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.axis('equal')
    plt.xlim(min_val - pad, max_val + pad)
    plt.ylim(min_val - pad, max_val + pad)
    plt.tight_layout()
    val_plot_path = os.path.join(save_dir, 'validation_plot.jpg')
    plt.savefig(val_plot_path, dpi=650, bbox_inches='tight')
    plt.savefig(val_plot_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()

    residuals = y_test_pred - y_test_true
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.7, edgecolor='k')
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.xlabel('Predicted Values', fontsize=24)
    plt.ylabel('Residuals', fontsize=24)
    plt.title('Residual Plot', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    res_plot_path = os.path.join(save_dir, 'residual_plot.jpg')
    plt.savefig(res_plot_path, dpi=650, bbox_inches='tight')
    plt.savefig(res_plot_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()

    if X_test.shape[1] == 2:
        abs_errors = np.abs(y_test_pred - y_test_true).flatten()
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(X_test[:, 0], X_test[:, 1], c=abs_errors,
                         cmap='viridis', alpha=0.8, edgecolor='k', s=50)
        plt.colorbar(sc, label='Absolute Error')
        plt.xlabel('Active Variable 1')
        plt.ylabel('Active Variable 2')
        plt.title('Prediction Error in 2-D Active Subspace')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residuals_2d_plot.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'residuals_2d_plot.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

    print(f"\nTest samples: {len(X_test)}")
    print(f"  R²   = {r2:.6f}")
    print(f"  RMSE = {rmse:.8f}")
    print(f"  MAE  = {mae:.8f}")
    print(f"Metrics saved to: {metrics_path}")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def compare_rom_fom_predictions(X, y_true, model,
                                 t_start=250.0, dt=0.05,
                                 qoi_label='$C_d$',
                                 geometry='cylinder',
                                 total_start_time=600.0,
                                 plot_start_time=750.0,
                                 plot_end_time=800.0,
                                 scatter_xlim=None,
                                 scatter_ylim=None,
                                 split_idx=None,
                                 save_dir='./results/ROM_FOM_Comparison'):
    """Compare ROM (response surface) and FOM (simulation) predictions.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Active-variable coordinates for all samples.
    y_true : ndarray, shape (N,)
        Ground-truth QoI values from the full-order model.
    model : object
        Response surface with a ``predict(X)`` method.
    t_start : float
        Physical start time (Cylinder only).
    dt : float
        Time step between consecutive samples.
    qoi_label : str
        LaTeX label for the QoI axis (e.g. ``'$C_l$'``).
    geometry : str
        ``'naca4412'`` – combined time-window+PDF figure and combined
        scatter+residuals figure (matches legacy Example2_NACA4412).
        ``'cylinder'`` – full time-series, separate scatter and residuals
        (matches legacy Example1_Cylinder).
    total_start_time : float
        Physical start time of the full dataset (NACA4412 = 600).
    plot_start_time : float
        Start of the displayed time window (NACA4412 = 750).
    plot_end_time : float
        End of the displayed time window (NACA4412 = 800).
    scatter_xlim : tuple, optional
        (xmin, xmax) for scatter/residuals axes. Auto-computed if None.
    scatter_ylim : tuple, optional
        (ymin, ymax) for scatter y-axis. Defaults to ``scatter_xlim``.
    save_dir : str
        Directory for saving figures.

    Returns
    -------
    dict
        Dictionary with keys ``'r2'``, ``'rmse'``, ``'mae'``.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X)[0].flatten()
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    print(f"ROM vs FOM: R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}")

    if geometry == 'naca4412':
        # ── Figure 1: time-window + normalised-deviation PDF ─────────────────
        fig = plt.figure(figsize=(20, 4))
        gs  = fig.add_gridspec(2, 4, hspace=0.0)
        ax1 = fig.add_subplot(gs[:, :3])

        s_idx = int((plot_start_time - total_start_time) / dt)
        e_idx = int((plot_end_time   - total_start_time) / dt)
        yt_sl = y_true[s_idx:e_idx]
        yp_sl = y_pred[s_idx:e_idx]
        t_sl  = plot_start_time + np.arange(len(yt_sl)) * dt

        if split_idx is not None:
            t_split_phys = total_start_time + split_idx * dt
            t_train_end  = min(t_split_phys, plot_end_time)
            t_pred_start = max(t_split_phys, plot_start_time)
            if t_train_end > plot_start_time:
                ax1.axvspan(plot_start_time, t_train_end, alpha=0.12, color='gray', zorder=0)
            if t_pred_start < plot_end_time:
                ax1.axvspan(t_pred_start, plot_end_time, alpha=0.12, color='steelblue', zorder=0)
            if plot_start_time <= t_split_phys <= plot_end_time:
                ax1.axvline(x=t_split_phys, color='gray', linestyle='--', linewidth=1.5)
                span = plot_end_time - plot_start_time
                y_top = float(yt_sl.max()) + 0.01 * (float(yt_sl.max()) - float(yt_sl.min()))


        # ax1.scatter(t_sl, yt_sl, color='black', s=50, alpha=0.7, label='FOM')
        ax1.plot(t_sl, yt_sl, 'k-', linewidth=2,  label='FOM')
        ax1.plot(t_sl, yp_sl, 'r--', linewidth=2, label='ROM')
        ax1.set_xlabel(r'$t$ /($CU_\infty^{-1}$)', fontsize=24)
        ax1.set_ylabel(qoi_label, fontsize=24)
        ax1.set_xlim([plot_start_time, plot_end_time])
        # ax1.legend(prop={'size': 24})

        inner  = qoi_label.strip('$')
        t_norm = np.abs(y_true - np.mean(y_true)) / np.std(y_true)
        p_norm = np.abs(y_pred - np.mean(y_pred)) / np.std(y_pred)
        bins   = np.linspace(0.0, 4, 50)

        if split_idx is not None:
            # ── Right panel: two stacked PDF subplots (train / test) ─────────
            ax2_top = fig.add_subplot(gs[0, 3])
            ax2_bot = fig.add_subplot(gs[1, 3], sharex=ax2_top)

            ax2_top.set_facecolor(mcolors.to_rgba('gray', alpha=0.12))
            ax2_bot.set_facecolor(mcolors.to_rgba('steelblue', alpha=0.12))

            for ax_p, t_n, p_n, add_leg in [
                (ax2_top, t_norm[:split_idx], p_norm[:split_idx], True),
                (ax2_bot, t_norm[split_idx:], p_norm[split_idx:], False),
            ]:
                h_t = np.histogram(t_n, density=True, bins=bins)
                h_p = np.histogram(p_n, density=True, bins=bins)
                ax_p.plot(h_t[1][:-1], h_t[0], 'k-',  linewidth=2, label='FOM')
                ax_p.plot(h_p[1][:-1], h_p[0], 'r--', linewidth=2, label='ROM')
                ax_p.set_yscale('log')
                ax_p.set_xlim([0, 3])
                ax_p.set_ylabel(r'$\rho$', fontsize=20)
                ax_p.tick_params(labelsize=24)
                # if add_leg:
                #     leg2 = ax_p.legend(prop={'size': 20}, frameon=False)
                #     leg2.get_frame().set_linewidth(0)
                #     leg2.get_frame().set_edgecolor('none')

            plt.setp(ax2_top.get_xticklabels(), visible=False)
            ax2_bot.set_xlabel(f'$|{inner}-\\mu|/\\sigma$', fontsize=22)
        else:
            ax2 = fig.add_subplot(gs[:, 3])
            t_hist = np.histogram(t_norm, density=True, bins=bins)
            p_hist = np.histogram(p_norm, density=True, bins=bins)
            ax2.plot(t_hist[1][:-1], t_hist[0], 'k-',  linewidth=2, label='FOM')
            ax2.plot(p_hist[1][:-1], p_hist[0], 'r--', linewidth=2, label='ROM')
            ax2.set_yscale('log')
            ax2.set_xlim([0, 3])
            ax2.set_xlabel(f'$|{inner}-\\mu|/\\sigma$', fontsize=24)
            ax2.set_ylabel(r'$\rho$', fontsize=24)
            leg2 = ax2.legend(prop={'size': 24})
            leg2.get_frame().set_linewidth(0)
            leg2.get_frame().set_edgecolor('none')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_curve.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_curve.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

        # ── Figure 2: scatter (FOM vs ROM) + residuals ───────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        min_v = min(float(y_true.min()), float(y_pred.min()))
        max_v = max(float(y_true.max()), float(y_pred.max()))
        pad   = (max_v - min_v) * 0.05

        if split_idx is not None:
            axes[0].scatter(y_true[:split_idx], y_pred[:split_idx],
                            alpha=0.5, edgecolor='k', color='steelblue', s=30, label='Training')
            axes[0].scatter(y_true[split_idx:], y_pred[split_idx:],
                            alpha=0.5, edgecolor='k', color='green', s=30, marker='^', label='Unseen Test')
            axes[0].legend(prop={'size': 20}, frameon=False)
        else:
            axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
        axes[0].plot([min_v - pad, max_v + pad],
                     [min_v - pad, max_v + pad], 'r--', lw=2)
        axes[0].set_xlabel('FOM', fontsize=26)
        axes[0].set_ylabel('ROM', fontsize=26)
        axes[0].set_xticks(axes[0].get_xticks())
        axes[0].set_yticks(axes[0].get_yticks())
        axes[0].tick_params(labelsize=24)
        xlim = scatter_xlim if scatter_xlim is not None else (min_v - pad, max_v + pad)
        ylim = scatter_ylim if scatter_ylim is not None else xlim
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_aspect('equal', adjustable='box')

        residuals = y_pred - y_true
        if split_idx is not None:
            axes[1].scatter(y_true[:split_idx], residuals[:split_idx],
                            alpha=0.5, edgecolor='k', color='steelblue', s=30, label='Training')
            axes[1].scatter(y_true[split_idx:], residuals[split_idx:],
                            alpha=0.5, edgecolor='k', color='green', s=30, marker='^', label='Unseen Test')
            axes[1].legend(prop={'size': 20}, frameon=False)
        else:
            axes[1].scatter(y_true, residuals, alpha=0.5, edgecolor='k')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel(qoi_label, fontsize=26)
        axes[1].set_ylabel('Residuals (ROM - FOM)', fontsize=26)
        axes[1].tick_params(labelsize=24)
        if scatter_xlim is not None:
            axes[1].set_xlim(scatter_xlim)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_residuals.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_residuals.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

    else:  # cylinder
        # ── Figure 1: full time series ───────────────────────────────────────
        time    = np.arange(y_true.shape[0]) * dt + t_start
        fig, ax = plt.subplots(figsize=(12, 6))

        if split_idx is not None:
            t_split = time[split_idx]
            ax.axvspan(time[0], t_split, alpha=0.08, color='gray')
            ax.axvspan(t_split, time[-1], alpha=0.08, color='steelblue')
            ax.axvline(x=t_split, color='gray', linestyle='--', linewidth=1.5)
            y_top = float(y_true.max()) + 0.002 * (float(y_true.max()) - float(y_true.min()))
            # ax.text((time[0] + t_split) / 2, y_top, 'Reconstruction',
            #         ha='center', va='bottom', fontsize=18, color='gray')
            # ax.text((t_split + time[-1]) / 2, y_top, 'Prediction',
                    # ha='center', va='bottom', fontsize=18, color='steelblue')

        ax.scatter(time, y_true, color='black', s=50, alpha=0.7, label='FOM')
        # ax.plot(time, y_true, 'k-', linewidth=2,  label='FOM')
        ax.plot(time, y_pred, 'r--', linewidth=2, label='ROM')
        ax.set_xlabel(r'$t/(DU_{\infty}^{-1})$', fontsize=26)
        ax.set_ylabel(qoi_label, fontsize=26)
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(labelsize=22)
        ax.legend(prop={'size': 24}, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_curve.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison_curve.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

        # ── Figure 2: scatter (FOM vs ROM) ───────────────────────────────────
        a_min = min(float(y_true.min()), float(y_pred.min()))
        a_max = max(float(y_true.max()), float(y_pred.max()))
        pad   = (a_max - a_min) * 0.05
        xlim  = scatter_xlim if scatter_xlim is not None else (a_min - pad, a_max + pad)
        ylim  = scatter_ylim if scatter_ylim is not None else xlim

        plt.figure(figsize=(8, 6))
        if split_idx is not None:
            plt.scatter(y_true[:split_idx], y_pred[:split_idx],
                        alpha=0.5, edgecolor='k', color='steelblue', s=30, label='Training')
            plt.scatter(y_true[split_idx:], y_pred[split_idx:],
                        alpha=0.5, edgecolor='k', color='green', s=30, marker='^', label='Unseen Test')
            plt.legend(prop={'size': 20}, frameon=False)
        else:
            plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'r--', lw=2)
        plt.xlabel('FOM', fontsize=26)
        plt.ylabel('ROM', fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'rom_fom_comparison.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

        # ── Figure 3: residuals ──────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        residuals = y_pred - y_true
        if split_idx is not None:
            plt.scatter(y_true[:split_idx], residuals[:split_idx],
                        alpha=0.5, edgecolor='k', color='steelblue', s=30)
            plt.scatter(y_true[split_idx:], residuals[split_idx:],
                        alpha=0.5, edgecolor='k', color='green', s=30, marker='^')
            # plt.legend(prop={'size': 20}, frameon=False)
        else:
            plt.scatter(y_true, residuals, alpha=0.7, edgecolor='k')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(qoi_label, fontsize=26)
        plt.ylabel('Residuals (ROM - FOM)', fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rom_fom_residuals.jpg'),
                    dpi=650, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'rom_fom_residuals.pdf'),
                    dpi=650, bbox_inches='tight')
        plt.close()

    # ── 2D error distribution (both geometries, only when X is 2-D) ─────────
    if X.shape[1] == 2:
        abs_err   = np.abs(y_pred - y_true).flatten()
        threshold = np.percentile(abs_err, 90)
        hi_mask   = abs_err > threshold

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(X[:, 0], X[:, 1], c=abs_err,
                         cmap='viridis', alpha=0.8, edgecolor='k', s=30)
        plt.scatter(X[hi_mask, 0], X[hi_mask, 1],
                    facecolor='none', edgecolor='red', s=100,
                    linewidth=2, label='High Error Points')
        cbar = plt.colorbar(sc)
        cbar.set_label('Absolute Error', fontsize=24)
        cbar.ax.tick_params(labelsize=24)
        plt.xlabel('Active Variable 1', fontsize=24)
        plt.ylabel('Active Variable 2', fontsize=24)
        plt.title('Error Distribution in 2D Active Subspace', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 24})
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution_2d.jpg'), dpi=650)
        plt.close()

    # ── Performance metrics ──────────────────────────────────────────────────
    performance_file = os.path.join(save_dir, 'performance_metrics.txt')
    with open(performance_file, 'w') as f:
        f.write('ROM vs FOM Performance Metrics\n')
        f.write('-' * 80 + '\n')
        f.write(f"R² Score: {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.8f}\n")
        f.write(f"MAE: {mae:.8f}\n")
        f.write('-' * 80 + '\n')
    print(f"Performance metrics saved to: {performance_file}")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}

def plot_subspace_polynomial_heatmap(XX_as, f, eigenvecs,
                                      n_dim_range=10, n_poly_range=10,
                                      save_dir='./results/Heatmap'):
    """R² heatmap over active-subspace dimension × polynomial order.

    For each (subspace dimension, polynomial order) pair the function trains a
    polynomial response surface on the first 80 % of samples (chronological
    split) and evaluates R² on the remaining 20 %.  Results are shown as a
    seaborn heatmap.

    Parameters
    ----------
    XX_as : ndarray, shape (N, d)
        Normalised POD coefficient inputs (already mapped to [-1, 1]).
    f : ndarray, shape (N, 1) or (N,)
        Target QoI values.
    eigenvecs : ndarray, shape (d, k)
        AS eigenvector matrix (rows = input dimensions, columns = AS directions).
    n_dim_range : int
        Maximum subspace dimension to test (1 … n_dim_range).
    n_poly_range : int
        Maximum polynomial order to test (1 … n_poly_range).
    save_dir : str
        Directory for saving the figure.

    Returns
    -------
    str
        Path to the saved heatmap figure.
    dict
        ``{'r2_matrix': ndarray}`` with shape ``(n_dim_range, n_poly_range)``.
    """
    import seaborn as sns
    from math import comb
    from sklearn.metrics import r2_score
    import lib.active_subspaces as ac

    os.makedirs(save_dir, exist_ok=True)

    r2_matrix = np.full((n_dim_range, n_poly_range), np.nan)

    n_total = XX_as.shape[0]
    split   = int(0.8 * n_total)

    for dim in range(1, n_dim_range + 1):
        y = XX_as.dot(eigenvecs[:, :dim])
        X_train, X_test = y[:split], y[split:]
        f_train, f_test = f[:split], f[split:]
        n_samples = X_train.shape[0]

        for poly_order in range(1, n_poly_range + 1):
            try:
                min_samples = comb(poly_order + dim, dim)
                if n_samples >= min_samples:
                    rs = ac.utils.rs.PolynomialApproximation(N=poly_order)
                    rs.train(X_train, f_train)
                    pred = rs.predict(X_test)[0]
                    r2_matrix[dim - 1, poly_order - 1] = r2_score(f_test, pred)
                else:
                    print(f"  Skipping dim={dim}, order={poly_order}: "
                          f"need {min_samples} samples, have {n_samples}")
            except Exception as exc:
                print(f"  Error at dim={dim}, order={poly_order}: {exc}")

    valid_vals = r2_matrix[~np.isnan(r2_matrix)]
    vmin_auto = float(valid_vals.min()) if valid_vals.size > 0 else 0.0
    vmax_auto = float(valid_vals.max()) if valid_vals.size > 0 else 1.0

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        r2_matrix,
        annot=True, fmt='.6f', cmap='Greys',
        annot_kws={"size": 32},
        xticklabels=range(1, n_poly_range + 1),
        yticklabels=range(1, n_dim_range + 1),
        vmin=vmin_auto-0.001, vmax=vmax_auto,
    )
    ax.tick_params(axis='both', labelsize=32)
    ax.collections[0].colorbar.ax.tick_params(labelsize=32)
    plt.xlabel('Degree of the response surface', fontsize=34)
    plt.ylabel('Active subspace dimension', fontsize=34)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'subspace_polynomial_heatmap.jpg')
    plt.savefig(plot_path, dpi=650, bbox_inches='tight')
    plt.savefig(plot_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()

    return plot_path, {'r2_matrix': r2_matrix}


def plot_interaction_heatmap(W, eigenvals, pod_importance, n_active,
                             top_n=6, save_dir='./results/Activity_Score',
                             use_scientific_colorbar=False, cbar_formula=None):
    """Lower-triangle modal interaction heatmap weighted by AS eigenvalues.

    Computes the interaction matrix
    ``M = sum_{i=1}^{n_active} lambda_i * w_i * w_i^T``
    and displays its top-left ``top_n × top_n`` sub-block as a lower-triangle
    colour map with manually drawn cell borders — matching the style of
    legacy ``main.py`` exactly.

    Parameters
    ----------
    W : ndarray, shape (num_pod_coeffs, num_pod_coeffs)
        Full AS eigenvector matrix (columns = directions).
    eigenvals : ndarray, shape (num_pod_coeffs, 1)
        Corresponding eigenvalues.
    pod_importance : ndarray, shape (num_pod_coeffs,)
        Normalised POD mode importance scores (used for tick labels).
    n_active : int
        Number of active AS directions to include in the sum.
    top_n : int
        Size of the sub-block to display (default 6).
    save_dir : str
        Directory for saving the figure.
    use_scientific_colorbar : bool
        If True, format colorbar ticks with scientific notation and move the
        power-of-ten offset label to the top of the colorbar — matches the
        NACA 4412 legacy style (default False).
    cbar_formula : str or None
        LaTeX string to annotate beside the colorbar.  Defaults to the
        Case 1 formula with hats when None.

    Returns
    -------
    str
        Path to the saved figure.
    """
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)

    if cbar_formula is None:
        cbar_formula = (
            r'$\sum_{i=1}^{n} \, \hat{\lambda}_i \,'
            r' (\hat{\boldsymbol{w}}_i \hat{\boldsymbol{w}}_i^T)$'
        )

    # Build interaction matrix
    num_pod_coeffs = W.shape[0]
    M = np.zeros((num_pod_coeffs, num_pod_coeffs))
    for i in range(n_active):
        w = W[:, i].reshape(-1, 1)
        M += eigenvals[i, 0] * (w @ w.T)

    M_top = M[:top_n, :top_n]

    # Upper-triangle mask (hidden)
    mask = np.triu(np.ones_like(M_top, dtype=bool), k=1)

    # Tick labels: top-n modes ranked by importance
    mode_indices = np.sort(np.argsort(pod_importance)[-top_n:])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        M_top, mask=mask, annot=False, fmt='.4f', cmap='bwr', center=0,
        xticklabels=[f'Mode{idx + 1}' for idx in mode_indices],
        yticklabels=[f'Mode{idx + 1}' for idx in mode_indices],
        ax=ax, cbar=True,
        linewidths=0, linecolor=None,
    )

    # Draw borders around visible (lower-triangle) cells
    n = M_top.shape[0]
    border_lw = 1.8
    border_color = 'black'
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            x, y = j, i
            up_exists    = (i + 1 < n) and (not mask[i + 1, j])
            down_exists  = (i - 1 >= 0) and (not mask[i - 1, j])
            left_exists  = (j - 1 >= 0) and (not mask[i, j - 1])
            right_exists = (j + 1 < n) and (not mask[i, j + 1])
            if not up_exists:
                ax.add_line(Line2D([x, x + 1], [y + 1, y + 1],
                                   linewidth=border_lw, color=border_color))
            if not down_exists:
                ax.add_line(Line2D([x, x + 1], [y, y],
                                   linewidth=border_lw, color=border_color))
            if not left_exists:
                ax.add_line(Line2D([x, x], [y, y + 1],
                                   linewidth=border_lw, color=border_color))
            if not right_exists:
                ax.add_line(Line2D([x + 1, x + 1], [y, y + 1],
                                   linewidth=border_lw, color=border_color))

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=26)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=26)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.tight_layout()
    colorbar = ax.collections[0].colorbar

    if use_scientific_colorbar:
        from matplotlib.ticker import ScalarFormatter
        colorbar.formatter = ScalarFormatter(useMathText=True)
        colorbar.formatter.set_scientific(True)
        colorbar.formatter.set_powerlimits((0, 0))
        colorbar.update_ticks()
        if hasattr(colorbar, 'ax') and hasattr(colorbar.ax, 'yaxis'):
            offset = colorbar.ax.yaxis.get_offset_text()
            offset.set_fontsize(22)
            offset.set_horizontalalignment('center')
            offset.set_verticalalignment('bottom')
            offset.set_position((0.5, 1.05))

    colorbar.ax.text(
        3.0, 0.5, cbar_formula,
        fontsize=26, rotation=90, va='center', ha='left',
        transform=colorbar.ax.transAxes,
    )

    figname = os.path.join(save_dir,
                           f'param_interaction_heatmap_lower_top{top_n}.jpg')
    plt.savefig(figname, dpi=650, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(figname.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight',
                pad_inches=0.0)
    plt.close()
    return figname


# ---------------------------------------------------------------------------
# POD visualisation helpers
# ---------------------------------------------------------------------------

def _get_cylinder_boundary(radius=0.5, center=(0, 0), n_points=100):
    """Return (x, y) arrays for the cylinder boundary circle."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def _get_wing_boundary(alpha=5, n_points=50):
    """Return (x, y) arrays for the NACA 4412 aerofoil boundary at angle *alpha* degrees."""
    m, p, t, c, x_nose = 0.04, 0.4, 0.12, 1, -0.25
    X, Y = [], []
    for j in range(n_points):
        x = j / (n_points - 1)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if x < p:
            yc  = m / p ** 2 * (2 * p * (x / c) - (x / c) ** 2)
            dyc = 2 * m / (1 - p) ** 2 * (p / c - x / c ** 2)
        else:
            yc  = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * (x / c) - (x / c) ** 2)
            dyc = 2 * m / (1 - p) ** 2 * (p / c - x / c ** 2)
        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta) + x_nose
        yu = yc + yt * np.cos(theta)
        xj = np.round(xu * np.cos(-alpha * np.pi / 180) + yu * np.sin(alpha * np.pi / 180), 5)
        yj = np.round(-xu * np.sin(alpha * np.pi / 180) + yu * np.cos(alpha * np.pi / 180), 5)
        X.append(xj); Y.append(yj)
    for j in range(n_points):
        x = 1 - (j + 1) / n_points
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if x < p:
            yc  = m / p ** 2 * (2 * p * (x / c) - (x / c) ** 2)
            dyc = 2 * m / (1 - p) ** 2 * (p / c - x / c ** 2)
        else:
            yc  = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * (x / c) - (x / c) ** 2)
            dyc = 2 * m / (1 - p) ** 2 * (p / c - x / c ** 2)
        theta = np.arctan(dyc)
        xb = x + yt * np.sin(theta) + x_nose
        yb = yc - yt * np.cos(theta)
        xj = np.round(xb * np.cos(-alpha * np.pi / 180) + yb * np.sin(alpha * np.pi / 180), 5)
        yj = np.round(-xb * np.sin(alpha * np.pi / 180) + yb * np.cos(alpha * np.pi / 180), 5)
        X.append(xj); Y.append(yj)
    return X, Y


def plot_pod_energy(Ds, num_modes=20, save_dir='./results/POD',
                    geometry='cylinder'):
    """Plot POD modal energy and cumulative energy on a dual-axis figure.

    Parameters
    ----------
    Ds : ndarray
        Eigenvalues (energy spectrum) from POD.
    num_modes : int
        Number of modes to display (default 20; use ``len(Ds)`` for all).
    save_dir : str
        Directory for saving output files.
    geometry : str
        ``'cylinder'`` or ``'naca4412'``.  Controls axis scaling and
        percentage conventions to match the respective legacy scripts.
    """
    import matplotlib.ticker as ticker

    os.makedirs(save_dir, exist_ok=True)
    lambda_vals = Ds.copy()
    num_plot = min(num_modes, len(lambda_vals))
    mode_indices = np.arange(1, num_plot + 1)
    total_energy = np.sum(lambda_vals)

    if geometry == 'cylinder':
        energy_pct = lambda_vals[:num_plot] / total_energy
        cumulative  = np.cumsum(energy_pct)
        threshold_values = [(90, r'90\%'), (99, r'99\%')]
        threshold_scale  = 100.0
        fixed_y_label    = 45
        x_log = False
        ylim2 = [40, 100]
        yticks2 = [40, 50, 60, 70, 80, 90, 100]
    else:
        energy_pct = lambda_vals[:num_plot] / total_energy * 100
        cumulative  = np.cumsum(energy_pct)
        threshold_values = [(90, r'90\%'), (99, r'99\%')]
        threshold_scale  = 1.0
        fixed_y_label    = 13
        x_log = True
        ylim2 = None
        yticks2 = None

    fig, ax1 = plt.subplots(figsize=(11, 8))

    color1 = 'tab:blue'
    ax1.set_xlabel('Modes', fontsize=28)
    ax1.set_ylabel(r'Energy (\%)', color=color1, fontsize=28)

    if geometry == 'cylinder':
        ax1.plot(mode_indices, energy_pct,
                 'o-', color=color1, markersize=12, linewidth=2,
                 markeredgewidth=2, clip_on=False, label='Modal energy')
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax1.plot(mode_indices, energy_pct,
                 '-', color=color1, linewidth=2, label='Modal energy')
        ax1.set_xscale('log')

    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    ax1.set_xlim([1, num_plot])

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel(r'Cumulative Energy (\%)', color=color2, fontsize=28)

    cum_plot = cumulative * threshold_scale if geometry == 'cylinder' else cumulative
    if geometry == 'cylinder':
        ax2.plot(mode_indices, cum_plot,
                 'x-', color=color2, markersize=12, linewidth=2,
                 markeredgewidth=2, clip_on=False, label='Cumulative energy')
    else:
        ax2.plot(mode_indices, cum_plot,
                 '-', color=color2, linewidth=2, label='Cumulative energy')
        ax2.set_xscale('log')

    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color2)

    if ylim2 is not None:
        ax2.set_ylim(ylim2)
    if yticks2 is not None:
        ax2.set_yticks(yticks2)
        ax2.set_yticklabels([str(y) for y in yticks2], color=color2)

    for threshold, label in threshold_values:
        check = cumulative * threshold_scale if geometry == 'cylinder' else cumulative
        if np.any(check >= threshold):
            k = np.argmax(check >= threshold) + 1
            ax2.axvline(x=k, color='gray', linestyle='--', alpha=1)
            ax2.text(k * 1.05, fixed_y_label,
                     f'{label}: $k={k}$',
                     fontsize=26, ha='left', va='bottom',
                     rotation=90, color='gray')

    ax1.grid(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if geometry == 'cylinder':
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='center right', fontsize=26,
                   bbox_to_anchor=(1, 0.6), frameon=False)
    else:
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='lower center', bbox_to_anchor=(0.3, 0.0),
                   fontsize=26, frameon=False)

    plt.tight_layout()

    base = os.path.join(save_dir, 'pod_energy')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'POD energy plot saved to {base}.[jpg|pdf]')


def plot_eigenvalues(S, num_values=20, save_dir='./results/POD', log_scale=True,
                     geometry='naca4412'):
    """Plot POD singular value spectrum.

    Parameters
    ----------
    S : ndarray
        Singular values from POD.
    num_values : int
        Number of values to display.
    save_dir : str
        Directory for saving output files.
    log_scale : bool
        If True (default), use log axes.
    geometry : str
        ``'cylinder'`` – semilogy, separate line+markers, LaTeX labels,
        x-tick every 2 (matches legacy Example1_Cylinder/visualize_pod.py).
        ``'naca4412'`` – full log-log, combined ``'ko-'``, plain labels
        (matches legacy Example2_NACA4412/visualize_pod.py).
    """
    from matplotlib.ticker import MultipleLocator

    os.makedirs(save_dir, exist_ok=True)
    num_values = min(num_values, len(S))
    indices = np.arange(1, num_values + 1)

    plt.figure(figsize=(10, 8))

    if geometry == 'cylinder':
        if log_scale:
            plt.semilogy(indices, S[:num_values], 'k-', linewidth=2)
            mark_indices = np.arange(1, num_values + 1, 1)
            if num_values not in mark_indices:
                mark_indices = np.append(mark_indices, num_values)
            for mark_idx in mark_indices:
                if mark_idx <= num_values:
                    plt.semilogy(mark_idx, S[mark_idx - 1], 'ko', markersize=12,
                                 clip_on=False, markerfacecolor='none',
                                 markeredgecolor='black', markeredgewidth=2, alpha=1)
            plt.xlim(1, num_values)
            plt.ylabel(r'$\mathrm{Eigenvalues}$', fontsize=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
        else:
            plt.plot(indices, S[:num_values], 'k-', linewidth=2)
            mark_indices = np.arange(1, num_values + 1, 1)
            if num_values not in mark_indices:
                mark_indices = np.append(mark_indices, num_values)
            for mark_idx in mark_indices:
                if mark_idx <= num_values:
                    plt.plot(mark_idx, S[mark_idx - 1], 'ko', markersize=12,
                             markerfacecolor='none', markeredgecolor='black',
                             markeredgewidth=2, alpha=0.7)
            plt.ylabel(r'$\mathrm{Eigenvalues}$', fontsize=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
        plt.xlabel(r'$\mathrm{Modes}$', fontsize=28)
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    else:
        plt.plot(indices, S[:num_values], 'ko-', markersize=12, clip_on=False,
                 markerfacecolor='none', markeredgecolor='black',
                 markeredgewidth=2, linewidth=2, alpha=1)
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel('Modes', fontsize=28)
        plt.ylabel('Eigenvalues', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlim([1, num_values])
        plt.grid(False)

    plt.tight_layout()
    base = os.path.join(save_dir, 'pod_eigenvalues_log')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'POD eigenvalue plot saved to {base}.[jpg|pdf]')


def plot_pod_modes_and_coeffs(PhiU, An, original_shape, x_grid, y_grid,
                               num_modes=6, geometry='cylinder',
                               region_mask=None, save_dir='./results/POD'):
    """Plot POD spatial modes (left column) and temporal coefficients (right column).

    Parameters
    ----------
    PhiU : ndarray, shape (n_spatial, n_modes)
        Spatial mode matrix.
    An : ndarray, shape (N, n_modes)
        Temporal coefficient matrix.
    original_shape : tuple
        (ny, nx) 2-D shape of the full vorticity grid.
    x_grid, y_grid : ndarray
        1-D coordinate arrays for the full spatial grid.
    num_modes : int
        Number of modes to display (default 6 for cylinder, 10 for NACA).
    geometry : str
        ``'cylinder'`` or ``'naca4412'``.
    region_mask : ndarray or None
        Boolean mask used to filter spatial points (NACA only).
    save_dir : str
        Directory for saving output files.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator
    import matplotlib.gridspec as gridspec

    os.makedirs(save_dir, exist_ok=True)
    num_modes = min(num_modes, PhiU.shape[1], An.shape[1])

    mode_data = []
    for i in range(num_modes):
        if region_mask is not None:
            full_mode = np.zeros(original_shape)
            full_mode[region_mask] = PhiU[:, i]
            mode_data.append(full_mode)
        else:
            try:
                mode_data.append(PhiU[:, i].reshape(original_shape))
            except ValueError:
                mode_data.append(np.zeros(original_shape))

    vmax = max(np.max(np.abs(m)) for m in mode_data)
    vmin = -vmax

    if geometry == 'cylinder':
        fig = plt.figure(figsize=(14, 20))
        gs  = gridspec.GridSpec(num_modes, 2, figure=fig,
                                width_ratios=[1, 1],
                                hspace=0.3, wspace=0.5)
        xlim, ylim   = (-2.0, 15.0), (-4.1, 4.1)
        fs_title     = 26
        fs_xy        = 32
        fs_tick      = 30
        fs_coeff_y   = 32
        fs_coeff_x   = 32
        fs_coeff_title = 32
        dt           = 0.05
        t_offset     = 0.0
        use_constrained = False
    else:
        fig = plt.figure(figsize=(15, 30), constrained_layout=True)
        gs  = gridspec.GridSpec(num_modes, 2, figure=fig,
                                width_ratios=[1, 1],
                                hspace=0.05, wspace=0.15)
        xlim, ylim   = (-0.5, 2.0), (-0.5, 0.5)
        fs_title     = 44
        fs_xy        = 46
        fs_tick      = 38
        fs_coeff_y   = 44
        fs_coeff_x   = 46
        fs_coeff_title = 44
        dt           = 0.04
        t_offset     = 600.0
        use_constrained = True

    time_arr = np.arange(An.shape[0]) * dt + t_offset
    extent   = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

    for i in range(num_modes):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(mode_data[i], cmap='bwr', vmin=vmin, vmax=vmax,
                       extent=extent, origin='lower', aspect='equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax.set_title(fr'$\boldsymbol{{\Phi}}_{{\omega, {i+1}}}$',
                     fontsize=fs_title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if geometry == 'cylinder':
            cx, cy = _get_cylinder_boundary(radius=0.5, center=(0, 0))
            ax.fill(cx, cy, color='k')
        else:
            wx, wy = _get_wing_boundary(alpha=5, n_points=200)
            ax.fill(wx, wy, color='k')

        ax.set_ylabel('$y$', fontsize=fs_xy)
        ax.tick_params(axis='both', labelsize=fs_tick)
        if i == num_modes - 1:
            ax.set_xlabel('$x$', fontsize=fs_xy)
        else:
            ax.set_xticks([])

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(time_arr, An[:, i], color='k', linewidth=2)
        ax2.set_xlim(time_arr[0], time_arr[-1])
        ax2.set_title(fr'$a_{{{i+1}}}(t)$', fontsize=fs_coeff_title)
        ax2.set_ylabel(r'$\boldsymbol{a}$', fontsize=fs_coeff_y)
        if geometry == 'naca4412':
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.tick_params(axis='both',
                        labelsize=(fs_tick if geometry == 'cylinder' else 36))
        if i == num_modes - 1:
            ax2.set_xlabel(r'$t$', fontsize=fs_coeff_x)
        else:
            ax2.set_xticks([])

    if not use_constrained:
        pass

    base = os.path.join(save_dir, 'pod_modes_and_coeffs')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'POD modes & coefficients plot saved to {base}.[jpg|pdf]')


def plot_pod_phase_space_triangle(An, num_modes=6, save_dir='./results/POD',
                                   geometry='cylinder'):
    """Lower-triangular phase-portrait matrix with diagonal KDE histograms.

    Parameters
    ----------
    An : ndarray, shape (N, n_modes)
        Temporal coefficient matrix.
    num_modes : int
        Number of leading modes to include.
    save_dir : str
        Directory for saving output files.
    geometry : str
        ``'cylinder'`` or ``'naca4412'``.
    """
    from scipy.stats import gaussian_kde
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator

    os.makedirs(save_dir, exist_ok=True)
    num_modes = min(num_modes, An.shape[1])

    if geometry == 'cylinder':
        figsize = (20, 20)
        fs_label = 28
        fs_tick  = 26
        fs_pdf   = 28
    else:
        figsize = (30, 30)
        fs_label = 40
        fs_tick  = 26
        fs_pdf   = 36

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(num_modes, num_modes, figure=fig,
                            hspace=0.4, wspace=0.4,
                            height_ratios=[1] * num_modes,
                            width_ratios=[1] * num_modes)

    for i in range(num_modes):
        for j in range(num_modes):
            ax = fig.add_subplot(gs[i, j])

            if j > i:
                ax.axis('off')
                continue

            if i == j:
                data = An[:, i]
                ax.hist(data, bins=30, density=True,
                        color='lightgray', edgecolor='k', alpha=0.6)
                kde    = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_vals, kde(x_vals), 'r-', lw=2)
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
                ax.set_ylabel('PDF', rotation=270, labelpad=20, fontsize=fs_pdf)
                if i == num_modes - 1:
                    ax.set_xlabel(f'$a_{{{i+1}}}$', fontsize=fs_label)
                ax.tick_params(axis='both', labelsize=fs_tick)
                ax.grid(True, alpha=0.3)
            else:
                ax.plot(An[:, j], An[:, i], 'k-', lw=1.5, alpha=0.7)
                if i == num_modes - 1:
                    ax.set_xlabel(f'$a_{{{j+1}}}$', fontsize=fs_label)
                if j == 0:
                    ax.set_ylabel(f'$a_{{{i+1}}}$', fontsize=fs_label)
                    if geometry == 'naca4412':
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.tick_params(axis='both', labelsize=fs_tick)

    base = os.path.join(save_dir, 'pod_phase_space_triangle')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'POD phase-space triangle plot saved to {base}.[jpg|pdf]')


def plot_mesh_and_vorticity(flow_data_path, geometry='cylinder',
                             nek_data_path=None, snapshot_idx=10,
                             save_dir='./results/Mesh'):
    """Plot the computational mesh (left) and a vorticity snapshot (right).

    The left panel requires raw Nek5000 field files (via *nek_data_path*) and
    the ``pymech`` library.  If unavailable the panel is skipped gracefully.
    The right panel always uses the pre-computed ``flow_field_data.npz``.

    Parameters
    ----------
    flow_data_path : str
        Path to ``flow_field_data.npz``.
    geometry : str
        ``'cylinder'`` or ``'naca4412'``.
    nek_data_path : str or None
        Path to the first Nek5000 snapshot file used for mesh extraction.
        If *None* or the file does not exist the mesh panel is omitted.
    snapshot_idx : int
        Index of the vorticity snapshot to display (default 10).
    save_dir : str
        Directory for saving output files.
    """
    from scipy.interpolate import RegularGridInterpolator

    os.makedirs(save_dir, exist_ok=True)

    flow_data   = np.load(flow_data_path)
    x_grid      = flow_data['vorticity_grid_x']
    y_grid      = flow_data['vorticity_grid_y']
    vort_data   = flow_data['vorticity']
    idx         = min(snapshot_idx, vort_data.shape[0] - 1)
    vort_raw    = vort_data[idx]

    if geometry == 'cylinder':
        nx_d, ny_d  = 593, 356
        scale       = 2
        mesh_xlim   = [-15, 35]
        mesh_ylim   = [-15, 15]
        vort_xlim   = [-15 / scale, 35 / scale]
        vort_ylim   = [-15 / scale, 15 / scale]
        vabs        = np.max(np.abs(vort_raw)) / 2
        vort_limit  = 1.0 if vabs < 1e-10 else min(vabs, 10.0)
        vmin_p, vmax_p = -vort_limit, vort_limit
    else:
        nx_d, ny_d  = 800, 400
        scale       = 2
        mesh_xlim   = [-2, 6.75]
        mesh_ylim   = [-2.5, 2.5]
        vort_xlim   = [-2 / scale, 6.75 / scale]
        vort_ylim   = [-2.5 / scale, 2.5 / scale]
        vmax_p      = np.max(vort_raw) / 30
        vmin_p      = -vmax_p

    x_disp = np.linspace(vort_xlim[0], vort_xlim[1], nx_d)
    y_disp = np.linspace(vort_ylim[0], vort_ylim[1], ny_d)
    XX, YY = np.meshgrid(x_disp, y_disp)

    interp_fn    = RegularGridInterpolator(
        (y_grid, x_grid), vort_raw,
        method='linear', bounds_error=False, fill_value=0.0,
    )
    vort_display = interp_fn(
        np.column_stack([YY.ravel(), XX.ravel()])
    ).reshape(ny_d, nx_d)

    label_fs  = 24
    has_mesh  = False
    field     = None
    nel       = 0

    if nek_data_path is not None and os.path.exists(nek_data_path):
        try:
            import pymech.neksuite as nek
            field    = nek.readnek(nek_data_path)
            nel      = len(field.elem)
            has_mesh = True
        except Exception as e:
            print(f'pymech unavailable or read error: {e}. Skipping mesh panel.')

    fig = plt.figure(figsize=(15, 7))

    if has_mesh:
        ax_mesh = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        for j in range(nel):
            x1 = field.elem[j].pos[0, 0, 0, 0];  x2 = field.elem[j].pos[0, 0, -1, 0]
            x3 = field.elem[j].pos[0, 0, -1, -1]; x4 = field.elem[j].pos[0, 0, 0, -1]
            y1 = field.elem[j].pos[1, 0, 0, 0];  y2 = field.elem[j].pos[1, 0, -1, 0]
            y3 = field.elem[j].pos[1, 0, -1, -1]; y4 = field.elem[j].pos[1, 0, 0, -1]
            ax_mesh.plot([x1, x2, x3, x4, x1],
                         [y1, y2, y3, y4, y1], 'k', linewidth=0.5)
        ax_mesh.set_xlim(mesh_xlim)
        ax_mesh.set_ylim(mesh_ylim)
        ax_mesh.set_aspect('equal')
        ax_mesh.tick_params(axis='both', labelsize=label_fs)
        ax_mesh.set_xlabel(r'$x$', fontsize=label_fs + 2)
        ax_mesh.set_ylabel(r'$y$', fontsize=label_fs + 2)
        ax_mesh.text(-0.08, 1.05, '(a)', transform=ax_mesh.transAxes,
                     fontsize=label_fs, fontweight='bold')
        ax_vort = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
    else:
        ax_vort = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=4)

    ax_vort.pcolor(XX, YY, vort_display, cmap='bwr',
                   vmin=vmin_p, vmax=vmax_p, shading='auto', rasterized=True)

    if geometry == 'cylinder':
        cx, cy = _get_cylinder_boundary(radius=0.5, center=(0, 0))
        ax_vort.fill(cx, cy, 'k')
    else:
        wx, wy = _get_wing_boundary(alpha=5, n_points=200)
        ax_vort.fill(wx, wy, 'k')

    ax_vort.set_xlim(vort_xlim)
    ax_vort.set_ylim(vort_ylim)
    ax_vort.set_aspect('equal')
    ax_vort.tick_params(axis='both', labelsize=label_fs)
    ax_vort.set_xlabel(r'$x$', fontsize=label_fs + 2)
    if not has_mesh:
        ax_vort.set_ylabel(r'$y$', fontsize=label_fs + 2)
    panel_b = '(b)' if has_mesh else '(a)'
    ax_vort.text(-0.08, 1.05, panel_b, transform=ax_vort.transAxes,
                 fontsize=label_fs, fontweight='bold')

    plt.tight_layout()
    base = os.path.join(save_dir, 'mesh')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'Mesh/vorticity plot saved to {base}.[jpg|pdf]')


def plot_qoi(qoi_data_path, qoi_label='$C_d$',
             time_label=r'$t$ /($DU_\infty^{-1}$)',
             save_dir='./results/QoI'):
    """Plot the raw QoI time series and its standardised smooth surrogate.

    Reproduces the two-panel layout of the legacy ``plot_qoi.py``:

    * Top panel   – raw QoI with ±2σ bounds.
    * Bottom panel – Gaussian-smoothed, zero-mean, unit-variance signal *q*
      with ±2σ bounds (matching the legacy style exactly, including the
      asymmetric bound line on the lower plot).

    The Gaussian smoothing bandwidth is derived from the dominant FFT
    frequency, consistent with the legacy implementation.

    Parameters
    ----------
    qoi_data_path : str
        Two-column (time, value) ``.dat`` or ``.npy`` file.
    qoi_label : str
        LaTeX y-axis label for the raw QoI (e.g. ``'$C_d$'``).
    time_label : str
        LaTeX x-axis label for the time axis.
    save_dir : str
        Directory for saving output files.
    """
    from scipy.ndimage import gaussian_filter1d

    os.makedirs(save_dir, exist_ok=True)

    if qoi_data_path.endswith('.npy'):
        fc = np.load(qoi_data_path)
    else:
        fc = np.loadtxt(qoi_data_path)
    qoi_time = fc[:, 0]
    qoi_vals = fc[:, 1]

    dt     = float(qoi_time[1] - qoi_time[0]) if len(qoi_time) > 1 else 1.0
    F_qoi  = np.fft.fft(qoi_vals - np.mean(qoi_vals))
    freqs  = np.fft.fftfreq(len(qoi_vals), d=dt)
    f_peak = freqs[np.argmax(np.abs(F_qoi))]
    print(f'f_peak: {f_peak:.6f}   period: {1.0 / f_peak if f_peak != 0 else float("inf"):.4f}')

    if f_peak != 0:
        scale_smoother = max(1, int(abs(0.5 / (f_peak * dt))))
    else:
        scale_smoother = 10

    q = gaussian_filter1d(qoi_vals, sigma=scale_smoother)
    print(f'Mean / std  q: {np.mean(q):.6f} / {np.std(q):.6f}')
    q = (q - np.mean(q)) / np.std(q)

    nsigma   = 2
    label_fs = 24
    t0, t1   = qoi_time[1], qoi_time[-1]

    fig = plt.figure(figsize=(15, 4))

    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax1.plot(qoi_time, qoi_vals, 'k', linewidth=1)
    mu, sg = np.mean(qoi_vals), np.std(qoi_vals)
    ax1.plot([t0, t1], [mu + nsigma * sg] * 2, 'r--', linewidth=1)
    ax1.plot([t0, t1], [mu - nsigma * sg] * 2, 'r--', linewidth=1)
    ax1.set_xlim([t0, t1])
    ax1.set_ylabel(qoi_label, fontsize=label_fs + 2)
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=label_fs)

    ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1)
    ax2.plot(qoi_time, q, 'k', linewidth=1)
    mq, sq = np.mean(q), np.std(q)
    ax2.plot([t0, t1], [mq + nsigma * sq] * 2, 'r--', linewidth=1)
    ax2.plot([t0, t1],
             [mq - nsigma * sq, mq + nsigma * sq], 'r--', linewidth=1)
    ax2.set_xlim([t0, t1])
    ax2.tick_params(axis='x', labelsize=label_fs)
    ax2.tick_params(axis='y', labelsize=label_fs)
    ax2.set_ylabel('$q$', fontsize=label_fs + 2)
    ax2.set_xlabel(time_label, fontsize=label_fs + 2)

    plt.tight_layout()
    base = os.path.join(save_dir, 'qoi')
    plt.savefig(base + '.jpg', dpi=650, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=650, bbox_inches='tight')
    plt.close()
    print(f'QoI plot saved to {base}.[jpg|pdf]')
