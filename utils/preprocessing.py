"""
Flow-field preprocessing utilities.

Merges multiple Nek5000 binary field files into a single ``flow_field_data.npz``
archive and computes the vorticity field on a regular Cartesian grid via linear
interpolation.

The optional ``geometry`` parameter handles geometry-specific masking:
- ``'none'``     – no masking (NACA 4412 style)
- ``'cylinder'`` – mask out the cylinder interior (Cylinder flow style)
"""

import os
import glob
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator


def extract_number(filename):
   
    match = re.search(r'f(\d+)', filename)
    return int(match.group(1)) if match else 0


def compute_vorticity_field(coords, velocity, resolution=(100, 100),
                             geometry='none'):
   
    x_flat = coords[:, 0, :].flatten()
    y_flat = coords[:, 1, :].flatten()
    u_flat = velocity[:, 0, :].flatten()
    v_flat = velocity[:, 1, :].flatten()

    points = np.column_stack((x_flat, y_flat))

    nx, ny = resolution
    x_min, x_max = x_flat.min(), x_flat.max()
    y_min, y_max = y_flat.min(), y_flat.max()
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    u_interp = LinearNDInterpolator(points, u_flat)(X, Y)
    v_interp = LinearNDInterpolator(points, v_flat)(X, Y)

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dvdx = np.gradient(v_interp, dx, axis=1)
    dudy = np.gradient(u_interp, dy, axis=0)
    vort = dvdx - dudy

    if geometry == 'cylinder':
        cx, cy, radius = 0.0, 0.0, 0.5
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < radius ** 2
        vort[mask] = np.nan

    return vort, x_grid, y_grid


def merge_flow_fields(
    data_dir,
    output_path='./merged_data/flow_field_data.npz',
    file_pattern='*.fld',
    resolution=(100, 100),
    geometry='none',
    max_files=None,
    reader_dir=None,
):
  
    if reader_dir is None:
        reader_dir = os.path.join(os.path.dirname(__file__), 'nek5000_reader')
    if reader_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(reader_dir))

    from nek5000_reader.fld_data import FldData

    files = sorted(
        glob.glob(os.path.join(data_dir, '**', file_pattern), recursive=True),
        key=extract_number,
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in '{data_dir}'"
        )
    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} field files to process.")

    vorticity_all = []
    time_stamps = []
    x_grid_ref = y_grid_ref = None

    for fpath in tqdm(files, desc="Processing field files"):
        try:
            fld = FldData.fromfile(fpath)
            coords = fld.coords
            velocity = fld.data
            time_stamps.append(fld.time)

            vort, x_grid, y_grid = compute_vorticity_field(
                coords, velocity, resolution=resolution, geometry=geometry
            )

            if x_grid_ref is None:
                x_grid_ref = x_grid
                y_grid_ref = y_grid

            vorticity_all.append(vort)
        except Exception as exc:
            print(f"Warning: skipping '{fpath}': {exc}")

    if not vorticity_all:
        raise RuntimeError("No vorticity fields were successfully computed.")

    vorticity_array = np.array(vorticity_all)
    print(f"Vorticity array shape: {vorticity_array.shape}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(
        output_path,
        vorticity=vorticity_array,
        time_stamps=np.array(time_stamps),
        vorticity_grid_x=x_grid_ref,
        vorticity_grid_y=y_grid_ref,
        vorticity_nx=resolution[0],
        vorticity_ny=resolution[1],
    )
    print(f"Merged data saved to: {output_path}")
    return os.path.abspath(output_path)


def plot_time_distribution(time_stamps, save_path=None):
   
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(time_stamps)), time_stamps, 'k-', linewidth=1.5)
    plt.xlabel('Snapshot index', fontsize=14)
    plt.ylabel('Physical time', fontsize=14)
    plt.title('Snapshot time distribution', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Time distribution plot saved to '{save_path}'")
    plt.show()
