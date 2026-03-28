# POD-ResNet-AS-PRS

> **A modular Python framework for data-driven reduced-order modelling of CFD flows**
>
> POD → ResNet → Active Subspaces → Polynomial Response Surface

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This repository provides a clean, modular implementation of the
**POD-ResNet-AS-PRS** workflow for building efficient, interpretable
surrogates of CFD quantity-of-interest (QoI) functionals directly from
high-fidelity flow-field snapshots.

| Step | Module | Method | Role |
|:----:|--------|--------|------|
| 1 | `core/pod_engine.py` | **Proper Orthogonal Decomposition** | Compress flow snapshots into a low-dimensional POD coefficient vector via truncated SVD |
| 2 | `core/resnet_model.py` · `resnet_trainer.py` | **Residual Network (ResNet)** | Learn the nonlinear map from POD coefficients to the scalar QoI |
| 3 | `core/gradient_analysis.py` | **Autograd / FD gradients** | Compute ∂QoI/∂POD by automatic differentiation; validate against central-difference FD |
| 4 | `lib/active_subspaces/` | **Active Subspaces (AS)** | Identify the dominant low-dimensional input subspace via eigendecomposition of the gradient covariance |
| 5 | `lib/.../utils/rs.py` | **Polynomial Response Surface (PRS)** | Fit a polynomial surrogate in the compressed active-variable space |

---

## Case Studies

Four fully reproducible Jupyter notebooks are provided:

| Notebook | Flow | QoI | Re | Notes |
|----------|------|-----|----|-------|
| `Case1_Cylinder.ipynb` | 2-D circular cylinder | $C_d$ | 100 | Active subspace dim = 2 |
| `Case1_Cylinder_Cl.ipynb` | 2-D circular cylinder | $C_l$ | 100 | Active subspace dim = 2; reuses Case 1 POD |
| `Case2_NACA4412.ipynb` | NACA 4412 aerofoil | $C_l$ | $1.75\times10^4$ | Active subspace dim = 22 |
| `Case2_NACA4412_Cd.ipynb` | NACA 4412 aerofoil | $C_d$ | $1.75\times10^4$ | Active subspace dim = 32; reuses Case 2 POD |

Each notebook covers the complete pipeline:
preprocessing → POD → ResNet training → gradient analysis → active subspaces → polynomial response surface → ROM vs FOM comparison.

---

## Repository Structure

```
POD-AS-PRS-Github-2/
├── core/                         # Core algorithmic modules
│   ├── pod_engine.py             #   POD via scipy truncated SVD
│   ├── resnet_model.py           #   Fully-connected ResNet (7 residual blocks)
│   ├── resnet_trainer.py         #   Training loop, early stopping, evaluation
│   └── gradient_analysis.py     #   Autograd & finite-difference gradient tools
│
├── lib/
│   └── active_subspaces/         # Active Subspaces library
│       ├── subspaces.py          #   Subspace computation (bootstrap CI)
│       ├── gradients.py          #   Local-linear / finite-difference gradients
│       └── utils/
│           ├── plotters.py       #   Unified plotting: eigenvalues, eigenvectors,
│           │                     #   heatmap, sufficient summary, zonotope
│           └── rs.py             #   Polynomial response surface (lstsq fit)
│
├── utils/                        # Shared utilities
│   ├── data_loader.py            #   Load snapshots, run POD, build DataLoaders
│   ├── preprocessing.py          #   Merge Nek5000 files, compute vorticity grid
│   ├── visualization.py          #   High-level figure helpers (ROM/FOM, heatmaps…)
│   └── nek5000_reader/           #   Binary Nek5000 field-file reader
│
├── examples/                     # Reproducible case-study notebooks
│   ├── Case1_Cylinder.ipynb      #   Cylinder Cd
│   ├── Case1_Cylinder_Cl.ipynb   #   Cylinder Cl
│   ├── Case2_NACA4412.ipynb      #   NACA 4412 Cl
│   └── Case2_NACA4412_Cd.ipynb   #   NACA 4412 Cd
│
├── data/                         # (not tracked) Raw and processed data files
│   ├── Case1_Cylinder/
│   └── Case2_NACA4412/
│
├── legacy/                       # Original monolithic scripts (reference only)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-org>/POD-AS-PRS.git
cd POD-AS-PRS

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

**Dependencies** — `numpy`, `scipy`, `matplotlib`, `torch>=2.0`,
`scikit-learn`, `pandas`, `seaborn`, `tqdm`, `jupyter`, `pymech`.

> **LaTeX** — A LaTeX distribution (TeX Live / MiKTeX) must be on `$PATH`
> because the plotting helpers use `matplotlib`'s `text.usetex = True` renderer.
> If LaTeX is unavailable, set `use_amsmath=False` in `plot_opts()`.

---

## Quick Start

### 1 · Preprocess raw Nek5000 snapshots *(one-off)*

```python
from utils.preprocessing import merge_flow_fields

merge_flow_fields(
    data_dir='data/Case1_Cylinder/raw/',
    output_path='data/Case1_Cylinder/flow_field_data.npz',
    geometry='cylinder',          # masks cylinder interior cells
)
```

### 2 · Run a case-study notebook

```bash
jupyter notebook examples/Case1_Cylinder.ipynb
```

### 3 · Use modules directly

```python
import torch
import numpy as np
from core.pod_engine       import POD_SVD
from core.resnet_model     import ResNet
from core.resnet_trainer   import set_random_seed, load_or_train, compute_all_gradients
from utils.data_loader     import load_and_preprocess_data, denormalise
import lib.active_subspaces as ac

# ── Data & POD ────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, pod_coeffs, pod_coeffs_norm, \
    pod_mean, pod_std, qoi_mean, qoi_std, *_ = load_and_preprocess_data(
        flow_data_path='data/Case1_Cylinder/flow_field_data.npz',
        qoi_data_path='data/Case1_Cylinder/drag_coefficient.dat',
        num_pod_coeffs=150,
    )

# ── ResNet surrogate ──────────────────────────────────────────────────────────
set_random_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = ResNet(input_size=150, hidden_size=128, num_blocks=7,
                dropout_rate=0.1).to(device)
model, train_losses, val_losses = load_or_train(
    model, train_loader, val_loader, device,
    model_save_path='results/resnet_model.pth',
)

# ── Gradients ─────────────────────────────────────────────────────────────────
gradients = compute_all_gradients(model, pod_coeffs, device, batch_size=32)

# ── Active Subspaces ──────────────────────────────────────────────────────────
XX_as_min, XX_as_max = pod_coeffs.min(axis=0), pod_coeffs.max(axis=0)
scale = (XX_as_max - XX_as_min) / 2.0
scale[scale < 1e-10] = 1.0
gradients_scaled = gradients * scale

ss = ac.subspaces.Subspaces()
ss.compute(df=gradients_scaled, nboot=1000)
ss.partition(2)

opts = ac.utils.plotters.plot_opts(savefigs=True)
ac.utils.plotters.eigenvalues(ss.eigenvals[:6], e_br=ss.e_br[:6], opts=opts)
ac.utils.plotters.eigenvectors(ss.eigenvecs[:6, :2], opts=opts)

# ── Polynomial Response Surface ───────────────────────────────────────────────
from lib.active_subspaces.utils.rs import PolynomialApproximation
from sklearn.model_selection import train_test_split

pod_norm_all = 2.0 * (pod_coeffs - XX_as_min) / (XX_as_max - XX_as_min) - 1.0
y_active     = pod_norm_all @ ss.W1           # (N, 2) active coordinates
qoi_raw      = np.loadtxt('data/Case1_Cylinder/drag_coefficient.dat')[:, 1]

X_tr, X_te, f_tr, f_te = train_test_split(y_active, qoi_raw, test_size=0.2,
                                            random_state=42)
RS = PolynomialApproximation(N=3)
RS.train(X_tr, f_tr.reshape(-1, 1))
print(f'RS train R² = {RS.Rsqr:.6f}')
```

---

## Key Design Choices

### Determinism
All notebooks call `set_random_seed(42)` **immediately before** model
construction to guarantee bit-exact reproducibility across runs.

### Gradient computation
`compute_all_gradients` uses **automatic differentiation** (PyTorch autograd)
for all production gradient matrices.  The finite-difference path in
`compare_gradients_nature_style_dataset` uses only the **training split**
(800 samples) to match the legacy benchmark timing.

### Unified `plotters.py`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_amsmath` | `True` | Load `\usepackage{amsmath}` in LaTeX preamble |
| `figsize` | `(8, 6)` | Per-function figure-size override |
| `sparse_xticks` | `False` | Show only even-indexed ticks (useful for 150-mode plots) |

The `eigenvectors_heatmap` function is available for both case studies.

### Spatial filtering
| Case | `apply_region_filter` | `geometry` | Effect |
|------|-----------------------|------------|--------|
| Cylinder | `False` | `'cylinder'` | Interior cells masked during preprocessing only |
| NACA 4412 | `True` | `'naca4412'` | Near-aerofoil spatial mask applied before POD |

---

## Outputs

Each case study generates the following artefacts under `results/<case>/`:

```
results/Case1_Cylinder/
├── POD/                   pod_energy, pod_modes_and_coeffs, pod_phase_space_triangle
├── Mesh/                  mesh, vorticity snapshot
├── loss_curve.[jpg|pdf]
├── metrics.txt            train / val / test MSE, MAE, R², MaxRelErr
├── gradient_comparison.pdf  AD vs FD violin plot
├── pod_gradients.npy
├── eigenvalues.[jpg|pdf]
├── eigenvectors.[jpg|pdf]
├── sufficient_summary.[jpg|pdf]
├── Importance/            POD mode importance bar chart
├── Heatmap/               subspace-dim × poly-order R² heatmap
├── Activity_Score/        modal interaction heatmap
├── Polynomial_CV/         cross-validation R² and RMSE curves
├── PRS/                   3-D response surface + contour
├── RS_Validation/         true-vs-predicted scatter + validation_metrics.txt
└── ROM_FOM/               time-series + scatter + performance_metrics.txt
```

---

## Acknowledgements

The `lib/active_subspaces/` directory is adapted from the
**Python Active Subspaces Utility Library** by Paul G. Constantine and
David Gleich, released under the MIT License.

> Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension
> Reduction in Parameter Studies*. SIAM Spotlights.
> doi: [10.1137/1.9781611973860](https://doi.org/10.1137/1.9781611973860)

Original repository: <https://github.com/paulcon/active_subspaces>

The following modifications were made relative to the original source:

- `subspaces.py` / `gradients.py` — translated inline comments to English;
  added English docstrings.
- `utils/plotters.py` — unified two per-case copies into a single file;
  added optional keyword arguments (`use_amsmath`, `figsize`, `sparse_xticks`);
  added `eigenvectors_heatmap()`.

All other files in `lib/active_subspaces/` are reproduced verbatim.
The original MIT license text is preserved in
`lib/active_subspaces/LICENSE`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2025,
  title   = {POD-ResNet-AS-PRS: A Modular Framework for
             Flow-Field Surrogate Modelling},
  author  = {Author, A. and Author, B.},
  journal = {Journal Name},
  year    = {2025},
  doi     = {10.xxxx/xxxxxx}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
