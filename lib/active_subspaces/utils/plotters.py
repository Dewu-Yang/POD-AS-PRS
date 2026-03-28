# Adapted from the Python Active Subspaces Utility Library
# Original: https://github.com/paulcon/active_subspaces
# Authors: Paul G. Constantine, David Gleich et al.
# License: MIT (see lib/active_subspaces/LICENSE)
# Modifications: unified two per-case copies into a single file; added
#   optional keyword arguments (use_amsmath, figsize, sparse_xticks) to
#   plot_opts(), eigenvalues(), eigenvectors(); added eigenvectors_heatmap();

"""Utilities for plotting quantities computed with active subspaces.

This module is the unified plotting back-end for all case studies.  All
per-case styling differences (figure size, LaTeX preamble, x-tick density,
etc.) are controlled via optional keyword arguments so that a single copy of
this file serves every example without duplication.

Public API
----------
plot_opts               Build a shared options dictionary.
eigenvalues             Semilog eigenvalue spectrum with bootstrap ranges.
subspace_errors         Semilog subspace-error plot with bootstrap ranges.
eigenvectors            Multi-eigenvector line plot (one line per vector).
eigenvectors_heatmap    Heat-map alternative for many eigenvectors / modes.
sufficient_summary      Scatter summary plot(s) in the active subspace.
zonotope_2d_plot        2-D zonotope with optional Delaunay triangulation.
show_plot               Thin wrapper around ``plt.show()``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, Delaunay, convex_hull_plot_2d, delaunay_plot_2d
import os

def plot_opts(savefigs=True, figtype='.jpg', use_amsmath=True):
    """Build a shared options dictionary for all plotting functions.

    Parameters
    ----------
    savefigs : bool
        If True, save figures to the ``figs/`` directory (default True).
    figtype : str
        File extension for saved images (default ``'.jpg'``).
    use_amsmath : bool
        If True, load the LaTeX ``amsmath`` package for math rendering
        (default True).  Set to False only when a minimal LaTeX installation
        lacks the package.

    Returns
    -------
    opts : dict
        Dictionary with keys ``'figtype'``, ``'savefigs'``, and ``'myfont'``.
    """
    if savefigs:
        if not os.path.isdir('figs'):
            os.mkdir('figs')

    myfont = {
        'family': 'serif',
        'size': 26,
        'weight': 'normal',
    }

    opts = {
        'figtype': figtype,
        'savefigs': savefigs,
        'myfont': myfont,
    }

    plt.rcParams['text.usetex'] = True
    if use_amsmath:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    return opts

def eigenvalues(e, e_br=None, out_label=None, opts=None, save_path=None,
                ylim=None, figsize=None, sparse_xticks=False):
    """Plot the eigenvalue spectrum with optional bootstrap confidence bands.

    Parameters
    ----------
    e : ndarray
        k-by-1 array of estimated eigenvalues.
    e_br : ndarray, optional
        k-by-2 array of bootstrap lower/upper bounds (default None).
    out_label : str, optional
        Label for the quantity of interest (default ``'Output'``).
    opts : dict, optional
        Options dictionary from :func:`plot_opts` (default None).
    save_path : str, optional
        Override the default save path.
    ylim : tuple, optional
        (ymin, ymax) axis limits.  When None the full data range is used.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches (default ``(8, 6)``).
    sparse_xticks : bool
        If True, show only every even index on the x-axis — useful when *k*
        is large (e.g. NACA4412 with 150 modes).  Default False.

    See Also
    --------
    utils.plotters.eigenvectors
    utils.plotters.subspace_errors
    """
    if opts is None:
        opts = plot_opts()
    if figsize is None:
        figsize = (8, 6)

    k = e.shape[0]
    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=figsize)
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1, k + 1), e, 'ko-', markersize=12, linewidth=2,
                 markeredgewidth=2, clip_on=False)
    if e_br is not None:
        plt.fill_between(range(1, k + 1), e_br[:, 0], e_br[:, 1],
                         facecolor='0.7', interpolate=True)
    plt.xlabel('Index', fontsize=26)
    plt.ylabel('Eigenvalues', fontsize=26)
    plt.grid(False)

    if sparse_xticks:
        xticks = [i for i in range(1, k + 1) if i % 2 == 0]
        plt.xticks(xticks, fontsize=24)
        plt.xlim(1, k)
    else:
        plt.xticks(range(1, k + 1), fontsize=24)

    plt.yticks(fontsize=24)
    plt.tight_layout()

    if ylim is not None:
        plt.axis([1, k, ylim[0], ylim[1]])

    if e_br is not None:
        indices = np.arange(1, k + 1).reshape(-1, 1)
        data_to_save = np.hstack((indices, e.reshape(-1, 1), e_br))
        header = 'Index\tEigenvalue\tLower_Bound\tUpper_Bound'
    else:
        indices = np.arange(1, k + 1).reshape(-1, 1)
        data_to_save = np.hstack((indices, e.reshape(-1, 1)))
        header = 'Index\tEigenvalue'

    np.savetxt('eigenval.dat', data_to_save, delimiter='\t',
               header=header, comments='')

    if opts['savefigs']:
        if save_path is None:
            save_path = f'figs/evals_{out_label}.jpg'
        plt.savefig(save_path, dpi=650, bbox_inches='tight')
        plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')

    plt.close()
    show_plot(plt)
def subspace_errors(sub_br ,out_label=None, opts=None):
    """Plot the estimated subspace errors with bootstrap ranges.

    Parameters
    ----------
    sub_br : ndarray
        (k-1)-by-3 matix that contains the lower bound, mean, and upper bound of
        the subspace errors for each dimension of subspace.
    out_label : str, optional 
        a label for the quantity of interest (default None)
    opts : dict, optional 
        a dictionary with some plot options (default None)

    See Also
    --------
    utils.plotters.eigenvectors
    utils.plotters.eigenvalues
    """
    if opts == None:
        opts = plot_opts()

    kk = sub_br.shape[0]
    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=(7,7))
    plt.rc('font', **opts['myfont'])
    plt.semilogy(range(1, kk+1), sub_br[:,1], 'ko-', markersize=12)
    plt.fill_between(range(1, kk+1), sub_br[:,0], sub_br[:,2],
        facecolor='0.7', interpolate=True)
    plt.xlabel('Subspace dimension')
    plt.ylabel('Subspace distance')
    plt.title(out_label)
    plt.grid(True)
    plt.xticks(range(1, kk+1))
    plt.axis([0, kk+1, 0.1*np.amin(sub_br[:,0]), 1])
    plt.tight_layout()

    if opts['savefigs']:
        figname = 'figs/subspace_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=650, bbox_inches='tight', pad_inches=0.0)

    show_plot(plt)
def eigenvectors(W, W_br=None, in_labels=None, out_label=None, opts=None,
                 save_path=None, figsize=None):
    """Plot all estimated eigenvectors on a single axes, one line per vector.

    Parameters
    ----------
    W : ndarray
        m-by-k matrix of k estimated eigenvectors (columns).
    W_br : ndarray, optional
        m-by-(2*k) matrix of bootstrap lower/upper bounds per eigenvector
        component (default None).
    in_labels : list of str, optional
        Labels for the m input dimensions (default None — use integer indices).
    out_label : str, optional
        Label for the quantity of interest (default ``'Output'``).
    opts : dict, optional
        Options dictionary from :func:`plot_opts` (default None).
    save_path : str, optional
        Override the default save path.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
        Default ``(8, 6)`` for few modes; consider ``(16, 6)`` for many modes.
    """
    if opts is None:
        opts = plot_opts()
    if figsize is None:
        figsize = (8, 6)

    m, n = W.shape

    if W_br is not None:
        _, n_br = W_br.shape
        if n_br != 2 * n:
            raise Exception(
                'Bootstrap range matrix must have 2*n columns (n_br=2n).'
            )

    if out_label is None:
        out_label = 'Output'

    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*',
               'X', '<', '>', 'P', 'H', '8', '1', '2', '3', '4', '+', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    plt.figure(figsize=figsize)
    plt.rc('font', **opts['myfont'])
    x_indices = range(1, m + 1)

    for i in range(n):
        plt.plot(
            x_indices, W[:, i],
            marker=markers[i % len(markers)],
            color='black',
            markersize=12,
            linewidth=2,
            clip_on=False,
            label=fr'$\boldsymbol{{w}}_{{{i+1}}}$',
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=2,
        )
        if W_br is not None:
            plt.fill_between(
                x_indices,
                W_br[:, 2 * i], W_br[:, 2 * i + 1],
                alpha=0.3,
                color=colors[i % len(colors)],
            )

    plt.xlabel('Modes', fontsize=26)
    plt.ylabel('Eigenvector components', fontsize=26)
    plt.grid(False)
    plt.xticks(x_indices, fontsize=24)
    plt.yticks(fontsize=24)

    if in_labels is not None:
        plt.xticks(x_indices, in_labels, rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2)

    plt.ylim(-0.1, 1.1)
    plt.xlim(1, m)
    plt.legend(fontsize=24, loc='best', frameon=False, fancybox=False, shadow=False)
    plt.tight_layout()

    indices = np.arange(1, m + 1).reshape(-1, 1)
    combined_data = indices
    header = 'Index'
    for i in range(n):
        combined_data = np.hstack((combined_data, W[:, i].reshape(-1, 1)))
        header += f'\tw{i+1}'
        if W_br is not None:
            combined_data = np.hstack((combined_data, W_br[:, 2 * i: 2 * i + 2]))
            header += f'\tw{i+1}_low\tw{i+1}_high'

    np.savetxt('eigenvec.dat', combined_data, delimiter='\t',
               header=header, comments='')

    if opts['savefigs']:
        if save_path is None:
            save_path = f'figs/evecs_{out_label}.jpg'
        plt.savefig(save_path, dpi=650, bbox_inches='tight')
        plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')

    plt.close()


def eigenvectors_heatmap(W, in_labels=None, out_label=None, opts=None,
                         save_path=None):
    """Plot eigenvector components as a colour heat-map.

    Rows correspond to eigenvectors ``w_1 … w_k``; columns correspond to
    input modes.  Useful when *m* (number of modes) is large.

    Parameters
    ----------
    W : ndarray
        m-by-k matrix of k estimated eigenvectors (columns).
    in_labels : list of str, optional
        Labels for the m input dimensions (default None).
    out_label : str, optional
        Label for the quantity of interest (default ``'Output'``).
    opts : dict, optional
        Options dictionary from :func:`plot_opts` (default None).
    save_path : str, optional
        Override the default save path.
    """
    if opts is None:
        opts = plot_opts()

    m, n = W.shape
    if out_label is None:
        out_label = 'Output'

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', **opts['myfont'])

    im = ax.imshow(W.T, aspect='auto', cmap='bwr', interpolation='nearest')

    if in_labels is not None:
        ax.set_xticks(np.arange(m))
        ax.set_xticklabels(
            [lbl if (i + 1) % 2 == 0 else '' for i, lbl in enumerate(in_labels)],
            rotation=45, ha='right', fontsize=28,
        )
    else:
        ax.set_xticks(np.arange(m))
        ax.set_xticklabels(
            [str(i + 1) if (i + 1) % 2 == 0 else '' for i in range(m)],
            fontsize=28,
        )

    y_labels = [fr'$\boldsymbol{{w}}_{{{i+1}}}$' for i in range(n)]
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(y_labels, fontsize=28)

    ax.set_xlabel('Modes', fontsize=28)
    ax.set_ylabel('Eigenvectors', fontsize=28)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.05, location='right')
    cbar.set_label('Component Value', fontsize=28)
    im.set_clim(-1, 1)
    cbar.ax.tick_params(labelsize=26)

    plt.tight_layout()

    if opts['savefigs']:
        if save_path is None:
            save_path = f'figs/evecs_heatmap_{out_label}.jpg'
        plt.savefig(save_path, dpi=650, bbox_inches='tight')
        plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')

    plt.close()

def sufficient_summary(y, f, out_label=None, opts=None, save_path=None):
    """Sufficient summary plot(s) in the active-variable space.

    Produces a 1-D scatter plot (``y`` vs ``f``) and, when ``y`` has two
    columns, also a 2-D colour scatter plot (active variable 1 vs 2, coloured
    by ``f``).

    Parameters
    ----------
    y : ndarray
        M-by-1 or M-by-2 matrix of active-variable coordinates.
    f : ndarray
        M-by-1 array of response values.
    out_label : str, optional
        Label for the response axis (default ``'Output'``).
    opts : dict, optional
        Options dictionary from :func:`plot_opts` (default None).
    save_path : str, optional
        Override the default save path for the 1-D plot.
    """
    if opts is None:
        opts = plot_opts()

    n = y.shape[1]
    if n == 1:
        y1 = y
    elif n == 2:
        y1 = y[:, 0]
        y2 = y[:, 1]
    else:
        raise Exception(
            'Sufficient summary plots cannot be made in more than 2 dimensions.'
        )

    if out_label is None:
        out_label = 'Output'

    plt.figure(figsize=(8, 6))
    plt.rc('font', **opts['myfont'])
    plt.scatter(y1.flatten(), f.flatten(), color='blue', alpha=0.7,
                s=60, edgecolors='k', linewidth=0.5)
    plt.xlabel(r'$\hat{\boldsymbol{W}}_1^T \boldsymbol{a}$', fontsize=26)
    plt.ylabel(out_label, fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(False)
    if opts['savefigs']:
        if save_path is None:
            save_path = f'figs/ssp1_{out_label}.jpg'
        plt.savefig(save_path, dpi=650, bbox_inches='tight')
        plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    show_plot(plt)
    plt.close()

    if n == 2:
        plt.figure(figsize=(8, 6))
        plt.rc('font', **opts['myfont'])
        plt.scatter(
            y1.flatten(), y2.flatten(),
            c=f.flatten(), s=80, alpha=0.8,
            edgecolors='k', linewidth=0.5,
            vmin=np.min(f), vmax=np.max(f),
            cmap='viridis',
        )
        plt.xlabel('Active variable 1', fontsize=26)
        plt.ylabel('Active variable 2', fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        x_min, x_max = np.min(y1), np.max(y1)
        y_min, y_max = np.min(y2), np.max(y2)
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.grid(False)
        plt.colorbar()
        plt.tight_layout()
        if opts['savefigs']:
            ssp2_path = f'figs/ssp2_{out_label}.jpg'
            plt.savefig(ssp2_path, dpi=650, bbox_inches='tight')
            plt.savefig(ssp2_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
        show_plot(plt)
        plt.close()
def zonotope_2d_plot(vertices, design=None, y=None, f=None, out_label=None, opts=None):
    """A utility for plotting (m,2) zonotopes with designs and quadrature rules.

    Parameters
    ----------
    vertices : ndarray 
        M-by-2 matrix that contains the vertices that define the zonotope
    design : ndarray, optional
        N-by-2 matrix that contains a design-of-experiments on the zonotope. The
        plot will contain the Delaunay triangulation of the points in `design` 
        and `vertices`. (default None)
    y : ndarray, optional 
        K-by-2 matrix that contains points to be plotted inside the zonotope. If
        `y` is given, then `f` must be given, too. (default None)
    f: ndarray, optional
        K-by-1 matrix that contains a color value for the associated points in 
        `y`. This is useful for plotting function values or quadrature rules 
        with the zonotope. If `f` is given, then `y` must be given, too. 
        (default None)
    out_label : str, optional 
        a label for the quantity of interest (default None)
    opts : dict, optional 
        a dictionary with some plot options (default None)

    Notes
    -----
    This function makes use of the scipy.spatial routines for plotting the
    zonotopes.
    """
    if opts == None:
        opts = plot_opts()

    # set labels for plots
    if out_label is None:
        out_label = 'Output'

    if vertices.shape[1] != 2:
        raise Exception('Zonotope vertices should be 2d.')

    if design is not None:
        if design.shape[1] != 2:
            raise Exception('Zonotope design should be 2d.')

    if y is not None:
        if y.shape[1] != 2:
            raise Exception('Zonotope design should be 2d.')

    if (y is not None and f is None) or (y is None and f is not None):
        raise Exception('You need both y and f to plot.')

    if y is not None and f is not None:
        if y.shape[0] != f.shape[0]:
            raise Exception('Lengths of y and f are not the same.')

    # get the xlim and ylim
    xmin, xmax = np.amin(vertices), np.amax(vertices)

    # make the Polygon patch for the zonotope
    ch = ConvexHull(vertices)

    # make the Delaunay triangulation
    if design is not None:
        points = np.vstack((design, vertices))
        dtri = Delaunay(points)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    fig0 = convex_hull_plot_2d(ch, ax=ax)
    for l in fig0.axes[0].get_children():
        if type(l) is Line2D:
            l.set_linewidth(3)

    if design is not None:
        fig1 = delaunay_plot_2d(dtri, ax=ax)
        for l in fig1.axes[0].get_children():
            if type(l) is Line2D:
                l.set_color('0.75')

    if y is not None:
        plt.scatter(y[:,0], y[:,1], c=f, s=100.0, vmin=np.min(f), vmax=np.max(f))
        plt.axes().set_aspect('equal')
        plt.title(out_label)
        plt.colorbar()

    plt.axis([1.1*xmin,1.1*xmax,1.1*xmin,1.1*xmax])
    plt.xlabel('Active variable 1')
    plt.ylabel('Active variable 2')
    show_plot(plt)
    if opts['savefigs']:
        figname = 'figs/zonotope_2d_' + out_label + opts['figtype']
        plt.savefig(figname, dpi=650, bbox_inches='tight', pad_inches=0.0)

def show_plot(plot, opts=None):
    plot.show()
