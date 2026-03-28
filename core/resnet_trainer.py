"""
Training, evaluation, and inference utilities for the POD-ResNet surrogate model.
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.switch_backend('Agg')

LABEL_SIZE = 24
FIG_WIDTH = 10
FIG_HEIGHT = 8

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


def set_random_seed(seed=42):
    """Fix all random-number generators for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def compute_metrics(y_true, y_pred):
    """Compute regression performance metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth values.
    y_pred : ndarray
        Predicted values.

    Returns
    -------
    mse, mae, r2, max_rel_error : float
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-8
    rel_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
    max_rel_error = np.max(rel_errors)
    return mse, mae, r2, max_rel_error


def evaluate_model(model, dataloader, device, criterion):
    """Compute average loss over a dataloader.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    device : torch.device
    criterion : loss function

    Returns
    -------
    float
        Mean loss value.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=1000,
    patience=100,
    lr=0.001,
    weight_decay=0.0,
    min_delta=0.0001,
    model_save_path='resnet_model.pth',
):
    """Train the ResNet model with early stopping.

    Parameters
    ----------
    model : nn.Module
        The ResNet model to train.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    device : torch.device
        Compute device.
    num_epochs : int
        Maximum number of training epochs (default 1000).
    patience : int
        Early-stopping patience in epochs (default 100).
    lr : float
        Initial Adam learning rate (default 0.001).
    weight_decay : float
        L2 regularisation coefficient for Adam (default 0.0).
        Set to 1e-4 for NACA 4412 to match legacy train.py line 149.
    min_delta : float
        Minimum improvement threshold for early stopping (default 0.0001).
    model_save_path : str
        File path for saving the best model checkpoint.

    Returns
    -------
    model : nn.Module
        Model restored to the best validation checkpoint.
    train_losses : list of float
    val_losses : list of float
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.5
    )

    patience_counter = 0
    best_val_loss = float('inf')
    best_model_weights = None
    train_losses = []
    val_losses = []

    print(f"\nStarting training (max {num_epochs} epochs, patience={patience})...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_loss = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()  # shallow copy — matches legacy train.py line 191
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Restored best model weights (val loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'")

    return model, train_losses, val_losses


def load_or_train(
    model,
    train_loader,
    val_loader,
    device,
    model_save_path,
    **train_kwargs,
):
    """Load an existing model checkpoint or train from scratch.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    device : torch.device
    model_save_path : str
        Path to look for (or save) the checkpoint.
    **train_kwargs :
        Additional keyword arguments forwarded to :func:`train`.

    Returns
    -------
    model : nn.Module
    train_losses, val_losses : list
        Empty lists when a checkpoint is loaded.
    """
    if os.path.exists(model_save_path):
        print(f"\nFound existing checkpoint: {model_save_path}. Loading...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            return model, [], []
        except RuntimeError as exc:
            print(f"Warning: checkpoint incompatible with current model ({exc}). "
                  "Discarding checkpoint and training from scratch.")
            os.remove(model_save_path)
            print(f"Removed stale checkpoint: {model_save_path}")

    return train(
        model,
        train_loader,
        val_loader,
        device,
        model_save_path=model_save_path,
        **train_kwargs,
    )


def compute_all_gradients(model, pod_coeffs, device, batch_size=32):
    """Compute autograd gradients for all samples in a POD coefficient matrix.

    Parameters
    ----------
    model : nn.Module
        Trained ResNet.
    pod_coeffs : ndarray
        Raw (un-normalised) POD coefficients, shape (N, n_modes).
    device : torch.device
    batch_size : int

    Returns
    -------
    ndarray
        Gradient matrix of shape (N, n_modes).
    """
    from .gradient_analysis import compute_gradients

    num_features = pod_coeffs.shape[1]

    pod_min = np.min(pod_coeffs, axis=0)
    pod_max = np.max(pod_coeffs, axis=0)
    pod_norm = 2.0 * (pod_coeffs - pod_min) / (pod_max - pod_min + 1e-12) - 1.0
    print(
        f"Normalised POD range: [{np.min(pod_norm):.4f}, {np.max(pod_norm):.4f}]"
    )

    all_pod_torch = torch.FloatTensor(pod_norm)
    dummy_targets = torch.zeros(all_pod_torch.shape[0], 1)

    num_samples = all_pod_torch.shape[0]
    gradients_all = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, num_samples)
        batch_pod = all_pod_torch[start:end].to(device)
        batch_tgt = dummy_targets[start:end].to(device)
        print(f"  Gradient batch {i+1}/{num_batches} (samples {start}–{end-1})")

        for j in range(end - start):
            s_pod = batch_pod[j: j + 1]
            if len(s_pod.shape) > 2:
                s_pod = s_pod.reshape(1, num_features)
            g = compute_gradients(model, s_pod, batch_tgt[j: j + 1], device)
            gradients_all.append(g.cpu().detach())

    autograd_gradients = np.array(gradients_all).reshape(num_samples, -1)
    return autograd_gradients


def evaluate_and_save_metrics(
    model, loaders, split_names, device, denorm_fn, results_dir='./results'
):
    """Evaluate the model on multiple splits and save a metrics text file.

    Parameters
    ----------
    model : nn.Module
    loaders : list of DataLoader
    split_names : list of str
        E.g. ['Train', 'Validation', 'Test'].
    device : torch.device
    denorm_fn : callable
        Function that converts normalised predictions back to physical units.
    results_dir : str

    Returns
    -------
    dict
        Mapping from split name to (mse, mae, r2, max_rel_error).
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    metrics = {}

    for split, loader in zip(split_names, loaders):
        all_inputs, all_targets = [], []
        for inputs, targets in loader:
            all_inputs.append(inputs)
            all_targets.append(targets)

        inputs_tensor = torch.cat(all_inputs, dim=0).to(device)
        targets_np = torch.cat(all_targets, dim=0).cpu().numpy()
        with torch.no_grad():
            preds_np = model(inputs_tensor).cpu().numpy()

        preds_np = denorm_fn(preds_np)
        targets_np = denorm_fn(targets_np)

        mse, mae, r2, max_rel = compute_metrics(targets_np, preds_np)
        metrics[split] = (mse, mae, r2, max_rel)
        print(
            f"{split}: MSE={mse:.6e}, MAE={mae:.6e}, R²={r2:.6f}, MaxRelErr={max_rel:.6f}"
        )

    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as fp:
        for split, (mse, mae, r2, max_rel) in metrics.items():
            fp.write(f"{split}:\n")
            fp.write(f"  MSE             = {mse:.12f}\n")
            fp.write(f"  MAE             = {mae:.12f}\n")
            fp.write(f"  R²              = {r2:.12f}\n")
            fp.write(f"  Max Rel. Error  = {max_rel:.12f}\n\n")
    print(f"Metrics saved to {metrics_path}")

    return metrics


def plot_loss_curve(train_losses, val_losses, patience, results_dir='./results'):
    """Save training/validation loss curve.

    Parameters
    ----------
    train_losses, val_losses : list of float
    patience : int
        Used to annotate the best-model and early-stopping points.
    results_dir : str
    """
    if not (train_losses and val_losses):
        print("No training history available; skipping loss curve.")
        return

    os.makedirs(results_dir, exist_ok=True)
    num_epochs = len(train_losses)

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(train_losses, label=r'$\mathrm{Train}$', linewidth=2)
    plt.plot(val_losses, label=r'$\mathrm{Validation}$', linewidth=2)

    if num_epochs < 1000:
        early_stop_epoch = num_epochs - 1
        best_epoch = max(0, early_stop_epoch - patience)
        plt.plot(
            best_epoch, val_losses[best_epoch],
            'y*', markersize=12, label=r'$\mathrm{Best\ model\ point}$',
        )
        plt.plot(
            early_stop_epoch, val_losses[early_stop_epoch],
            'r.', markersize=12, label=r'$\mathrm{Early\ stopping\ point}$',
        )

    plt.xlabel(r'$\mathrm{Epochs}$', fontsize=LABEL_SIZE + 2)
    plt.ylabel(r'$\mathrm{Loss}$', fontsize=LABEL_SIZE + 2)
    plt.xticks(fontsize=LABEL_SIZE)
    plt.yticks(fontsize=LABEL_SIZE)
    plt.yscale('log')
    plt.legend(fontsize=LABEL_SIZE, frameon=False)

    save_path = os.path.join(results_dir, 'loss_curve.jpg')
    plt.savefig(save_path, dpi=650, bbox_inches='tight')
    plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to '{save_path}'")


def plot_prediction_comparison(
    y_true, y_pred, dt=0.05, t_start=0.0, qoi_label='$C_d$', results_dir='./results'
):
    """Save a two-panel prediction comparison figure (time series + scatter).

    Parameters
    ----------
    y_true, y_pred : ndarray
        De-normalised ground-truth and predicted QoI values.
    dt : float
        Time step between consecutive samples.
    t_start : float
        Physical start time of the test sequence.
    qoi_label : str
        LaTeX label for the QoI (e.g. ``'$C_d$'``).
    results_dir : str
    """
    os.makedirs(results_dir, exist_ok=True)
    time_arr = np.arange(y_true.shape[0]) * dt + t_start

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    axes[0].plot(time_arr, y_true, color='red', linestyle='--',
                 label='Ground truth', linewidth=2)
    axes[0].plot(time_arr, y_pred, label='Predicted', marker='o',
                 color='black', linestyle='None', alpha=0.7,
                 markeredgecolor='k', markersize=8)
    axes[0].set_xlabel('$t$', fontsize=LABEL_SIZE + 2)
    axes[0].set_ylabel(qoi_label, fontsize=LABEL_SIZE + 2)
    axes[0].tick_params(axis='both', labelsize=LABEL_SIZE)
    axes[0].legend(fontsize=LABEL_SIZE, loc='lower right',
                   bbox_to_anchor=(0.8, 0.06), frameon=False)
    axes[0].set_xlim(time_arr[0], time_arr[-1])

    axes[1].scatter(y_true, y_pred, alpha=0.7, edgecolor='k', s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1].set_xlabel('Ground truth', fontsize=LABEL_SIZE + 2)
    axes[1].set_ylabel('Predicted', fontsize=LABEL_SIZE + 2)
    axes[1].tick_params(axis='both', labelsize=LABEL_SIZE)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.25)

    save_path = os.path.join(results_dir, 'prediction_comparison.jpg')
    plt.savefig(save_path, dpi=650, bbox_inches='tight')
    plt.savefig(save_path.replace('.jpg', '.pdf'), dpi=650, bbox_inches='tight')
    plt.close()
    print(f"Prediction comparison saved to '{save_path}'")
