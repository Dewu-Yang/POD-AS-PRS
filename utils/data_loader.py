"""
Data loading and preprocessing pipeline for the POD-ResNet surrogate.

Supports two flow configurations out of the box:
- Cylinder flow  (no spatial mask)
- NACA 4412 airfoil  (optional spatial-region mask)

The two cases are handled by a single :func:`load_and_preprocess_data`
function that accepts an ``apply_region_filter`` flag.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def set_random_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _run_pod(vort_reshaped, pod_save_dir, flow_data):
  
    from core.pod_engine import POD_SVD

    os.makedirs(pod_save_dir, exist_ok=True)
    pod_data_file = os.path.join(pod_save_dir, 'pod_data.npz')

    if os.path.exists(pod_data_file):
        print(f"Loading cached POD data from: {pod_data_file}")
        pod_data = np.load(pod_data_file)
        return (
            pod_data['U_mean'],
            pod_data['An'],
            pod_data['Phi'],
            pod_data['Ds'],
            pod_data['S'],
        )

    print("No cached POD data found. Running POD decomposition...")
    U0x, An, PhiU, Ds, S = POD_SVD(vort_reshaped)

    save_kwargs = dict(U_mean=U0x, An=An, Phi=PhiU, S=S, Ds=Ds)
    for key in ('vorticity_grid_x', 'vorticity_grid_y',
                'vorticity_nx', 'vorticity_ny'):
        if key in flow_data:
            save_kwargs[key] = flow_data[key]

    np.savez(pod_data_file, **save_kwargs)
    print(f"POD data saved to: {pod_data_file}")
    return U0x, An, PhiU, Ds, S


def _load_vorticity(flow_data_path, apply_region_filter, pod_save_dir):
    
    pod_data_file = os.path.join(pod_save_dir, 'pod_data.npz')
    if os.path.exists(pod_data_file):
        print(f"Found cached POD data — skipping flow field load: {pod_data_file}")
        pod_data = np.load(pod_data_file)
        return (
            pod_data['U_mean'],
            pod_data['An'],
            pod_data['Phi'],
            pod_data['Ds'],
            pod_data['S'],
        )

    print("\nLoading flow field data...")
    flow_data = np.load(flow_data_path)
    print("Flow data keys:", list(flow_data.keys()))

    vort_data = flow_data['vorticity']
    print(f"Vorticity array shape: {vort_data.shape}")

    x_grid = flow_data['vorticity_grid_x']
    y_grid = flow_data['vorticity_grid_y']
    X, Y = np.meshgrid(x_grid, y_grid)
    print(f"Grid range: X=[{X.min():.2f}, {X.max():.2f}], Y=[{Y.min():.2f}, {Y.max():.2f}]")

    if apply_region_filter:
        mask = (
            (X >= X.min()) & (X <= X.max()) &
            (Y >= Y.min()) & (Y <= Y.max())
        )
        print(f"Spatial filter: {np.sum(mask)} / {mask.size} points retained "
              f"({np.sum(mask) / mask.size * 100:.2f}%)")

        os.makedirs(pod_save_dir, exist_ok=True)
        np.save(os.path.join(pod_save_dir, 'region_mask.npy'), mask)

        n_timesteps = vort_data.shape[0]
        vort_reshaped = np.zeros((n_timesteps, int(np.sum(mask))))
        for t in range(n_timesteps):
            vort_reshaped[t] = vort_data[t][mask]
        print(f"Filtered vorticity shape: {vort_reshaped.shape}")
    else:
        n_timesteps, nx, ny = vort_data.shape
        vort_reshaped = vort_data.reshape(n_timesteps, nx * ny)
        print(f"Reshaped vorticity: {vort_reshaped.shape}")

    return _run_pod(vort_reshaped, pod_save_dir, flow_data)


def _normalise_split(array, mean, std):
    """Z-score normalise *array* using pre-computed *mean* and *std*."""
    normed = (array - mean) / std
    return np.nan_to_num(normed)


def load_and_preprocess_data(
    flow_data_path,
    qoi_data_path,
    num_pod_coeffs=100,
    train_ratio=0.8,
    val_ratio=0.1,
    batch_size=32,
    seed=42,
    shuffle_train=True,
    apply_region_filter=False,
    pod_save_dir='./results/POD',
):
   
    set_random_seed(seed)

    # ---------- POD decomposition ----------
    _, An, _, _, _ = _load_vorticity(flow_data_path, apply_region_filter, pod_save_dir)

    # ---------- QoI loading ----------
    print(f"\nLoading QoI data from: {qoi_data_path}")
    if qoi_data_path.endswith('.npy'):
        qoi_raw = np.load(qoi_data_path)
        qoi_vals = qoi_raw[:, 1].reshape(-1, 1)
    else:
        qoi_raw = np.loadtxt(qoi_data_path)
        qoi_vals = qoi_raw[:, 1].reshape(-1, 1)

    print(f"QoI shape: {qoi_vals.shape}, range: [{qoi_vals.min():.6f}, {qoi_vals.max():.6f}]")

    # ---------- Align lengths ----------
    min_length = min(An.shape[0], qoi_vals.shape[0])
    pod_coeffs = An[:min_length, :num_pod_coeffs]
    qoi_vals = qoi_vals[:min_length]
    print(f"POD coefficients used: {pod_coeffs.shape}")
    print(f"QoI values used:       {qoi_vals.shape}")

    # ---------- Temporal split (no shuffle before split) ----------
    split_train = int(train_ratio * min_length)
    split_val = int((train_ratio + val_ratio) * min_length)

    pod_train = pod_coeffs[:split_train]
    pod_val   = pod_coeffs[split_train:split_val]
    pod_test  = pod_coeffs[split_val:]
    qoi_train = qoi_vals[:split_train]
    qoi_val   = qoi_vals[split_train:split_val]
    qoi_test  = qoi_vals[split_val:]

    # ---------- Compute normalisation statistics on the training set only ----------
    pod_min = np.min(pod_train, axis=0)
    pod_max = np.max(pod_train, axis=0)
    pod_mean = np.mean(pod_train, axis=0)
    pod_std  = np.std(pod_train, axis=0)
    pod_std[pod_std < 1e-8] = 1.0

    qoi_min  = float(np.min(qoi_train))
    qoi_max  = float(np.max(qoi_train))
    qoi_mean = float(np.mean(qoi_train))
    qoi_std  = float(np.std(qoi_train))
    if qoi_std < 1e-8:
        qoi_std = 1.0

    # ---------- Z-score normalisation ----------
    pod_train_n = _normalise_split(pod_train, pod_mean, pod_std)
    pod_val_n   = _normalise_split(pod_val,   pod_mean, pod_std)
    pod_test_n  = _normalise_split(pod_test,  pod_mean, pod_std)
    qoi_train_n = _normalise_split(qoi_train, qoi_mean, qoi_std)
    qoi_val_n   = _normalise_split(qoi_val,   qoi_mean, qoi_std)
    qoi_test_n  = _normalise_split(qoi_test,  qoi_mean, qoi_std)

    # ---------- Reshape for ResNet ----------
    if num_pod_coeffs == 100:
        reshape_dims = (-1, 1, 10, 10)
    else:
        reshape_dims = (-1, 1, 1, num_pod_coeffs)

    def _to_tensor_loader(x, y, shuffle):
        x_t = torch.FloatTensor(x).reshape(reshape_dims)
        y_t = torch.FloatTensor(y)
        ds = TensorDataset(x_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_tensor_loader(pod_train_n, qoi_train_n, shuffle_train)
    val_loader   = _to_tensor_loader(pod_val_n,   qoi_val_n,   False)
    test_loader  = _to_tensor_loader(pod_test_n,  qoi_test_n,  False)

    pod_coeffs_norm = np.vstack([pod_train_n, pod_val_n, pod_test_n])

    return (
        train_loader, val_loader, test_loader,
        pod_coeffs, pod_coeffs_norm,
        pod_mean, pod_std,
        qoi_mean, qoi_std,
        pod_min, pod_max,
        qoi_min, qoi_max,
    )


def load_pod_vis_data(pod_save_dir, flow_data_path, apply_region_filter=False):
    
    pod_data_file = os.path.join(pod_save_dir, 'pod_data.npz')
    if not os.path.exists(pod_data_file):
        raise FileNotFoundError(
            f"POD cache not found: {pod_data_file}\n"
            "Run load_and_preprocess_data() first."
        )

    pod_data = np.load(pod_data_file)
    PhiU = pod_data['Phi']
    An   = pod_data['An']
    Ds   = pod_data['Ds']
    S    = pod_data['S']
    x_grid = pod_data['vorticity_grid_x'] if 'vorticity_grid_x' in pod_data else None
    y_grid = pod_data['vorticity_grid_y'] if 'vorticity_grid_y' in pod_data else None

    flow_data = np.load(flow_data_path)
    vort_shape = flow_data['vorticity'].shape
    original_shape = vort_shape[1:]

    if x_grid is None:
        nx, ny = vort_shape[2], vort_shape[1]
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)

    region_mask = None
    mask_file = os.path.join(pod_save_dir, 'region_mask.npy')
    if apply_region_filter and os.path.exists(mask_file):
        region_mask = np.load(mask_file)

    return dict(PhiU=PhiU, An=An, Ds=Ds, S=S,
                x_grid=x_grid, y_grid=y_grid,
                original_shape=original_shape,
                region_mask=region_mask)


def denormalise(values, mean, std):

    return values * std + mean
