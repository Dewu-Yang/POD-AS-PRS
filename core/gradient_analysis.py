import time
import os

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_SIZE = 24


def compute_gradients(model, pod_coeffs, target_qoi=None, device='cuda',
                      measure_time=False):
   
    pod_coeffs = pod_coeffs.clone().to(device)
    pod_coeffs.requires_grad_(True)
    model.eval()

    start = time.time()
    output = model(pod_coeffs)
    if output.numel() > 1:
        output = output[0]

    gradients = autograd.grad(outputs=output, inputs=pod_coeffs,
                              create_graph=False)[0]
    elapsed = time.time() - start

    if measure_time:
        return gradients, elapsed
    return gradients


def _process_inputs(X):
    X = X.reshape(X.shape[0], -1)
    M, m = X.shape
    return X, M, m


def _central_difference(X, fun, h=1e-7):
   
    X_flat = X.reshape(1, -1)
    m = X_flat.shape[1]
    grad = np.zeros_like(X_flat)

    for i in range(m):
        xp, xm = X_flat[0].copy(), X_flat[0].copy()
        xp[i] += h
        xm[i] -= h
        grad[0, i] = (fun(xp) - fun(xm)) / (2.0 * h)

    return grad.reshape(X.shape)


def compute_finite_diff_gradients(model, pod_coeffs, device, h=1e-7,
                                   measure_time=False):
    
    model.eval()
    X_np = pod_coeffs.clone().detach().cpu().numpy()
    input_size = model.input_layer[0].in_features

    def model_predict(x):
        x = np.array(x).flatten()
        if len(x) > input_size:
            x = x[:input_size]
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return model(x_t).cpu().numpy().flatten()[0]

    start = time.time()
    grad = _central_difference(X_np, model_predict, h)
    elapsed = time.time() - start
    grad = grad.reshape(X_np.shape)

    if measure_time:
        return grad, elapsed
    return grad


def compare_gradients(model, pod_coeffs, target_qoi=None, device='cuda',
                      h=1e-7, max_modes=20, save_path=None):

    print("\nComputing autograd gradients...")
    ad_grad = compute_gradients(model, pod_coeffs, target_qoi, device)
    ad_flat = ad_grad.cpu().detach().numpy().reshape(-1)[:max_modes]

    print("Computing finite-difference gradients...")
    fd_grad = compute_finite_diff_gradients(model, pod_coeffs, device, h)
    fd_flat = fd_grad.reshape(-1)[:max_modes]

    eps = 1e-5
    rel_diff = np.minimum(np.abs(ad_flat - fd_flat) / (np.abs(fd_flat) + eps), 100.0)
    avg_rel_diff = np.mean(rel_diff)
    max_rel_diff = np.max(rel_diff)

    print(f"\nAD vs FD comparison:")
    print(f"  Mean relative error: {avg_rel_diff:.6f}")
    print(f"  Max  relative error: {max_rel_diff:.6f}")

    x = np.arange(1, max_modes + 1)
    width = 0.35
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.bar(x - width / 2, ad_flat, width, label='AD')
    plt.bar(x + width / 2, fd_flat, width, label='Finite Difference')
    plt.xlabel('POD Mode', fontsize=24)
    plt.ylabel('Gradient Value', fontsize=24)
    plt.title('Gradient Comparison: Autograd vs Finite Difference', fontsize=24)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.bar(x, rel_diff)
    plt.xlabel('POD Mode', fontsize=24)
    plt.ylabel('Relative Difference', fontsize=24)
    plt.title(
        f'Relative Difference (Avg: {avg_rel_diff:.4f}, Max: {max_rel_diff:.4f})',
        fontsize=24,
    )
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = './results/gradient_comparison.jpg'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gradient comparison plot saved to '{save_path}'")

    return avg_rel_diff, max_rel_diff


def compare_gradients_nature_style_dataset(
    model, pod_coeffs, target_qoi=None, device='cuda',
    h=1e-2, max_modes=20, save_path=None, batch_size=32
):
   
    model.eval()
    if isinstance(pod_coeffs, np.ndarray):
        pod_coeffs = torch.FloatTensor(pod_coeffs)

    num_samples = pod_coeffs.shape[0]
    print(f"\nDataset-level gradient comparison for {num_samples} samples...")

    all_ad, all_fd = [], []
    ad_times, fd_times = [], []
    total_ad_time = total_fd_time = 0.0

    num_batches = (num_samples + batch_size - 1) // batch_size
    overall_start = time.time()

    for bi in range(num_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, num_samples)
        print(f"  Batch {bi+1}/{num_batches} (samples {s}–{e-1})")
        batch_pod = pod_coeffs[s:e]
        batch_tgt = None if target_qoi is None else target_qoi[s:e]

        for i in range(batch_pod.shape[0]):
            sp = batch_pod[i: i + 1].to(device)
            st = None if batch_tgt is None else batch_tgt[i: i + 1].to(device)

            ag, at = compute_gradients(model, sp, st, device, measure_time=True)
            ag = ag.cpu().detach().numpy().reshape(-1)[:max_modes]
            total_ad_time += at
            ad_times.append(at)

            fg, ft = compute_finite_diff_gradients(model, sp, device, h,
                                                       measure_time=True)
            fg = fg.reshape(-1)[:max_modes]
            total_fd_time += ft
            fd_times.append(ft)

            all_ad.append(ag)
            all_fd.append(fg)

    total_computation_time = time.time() - overall_start
    all_ad = np.array(all_ad)
    all_fd = np.array(all_fd)

    avg_ad_time = np.mean(ad_times)
    avg_fd_time = np.mean(fd_times)
    std_ad_time = np.std(ad_times)
    std_fd_time = np.std(fd_times)
    speedup_ratio = total_fd_time / (total_ad_time + 1e-12)

    rel_diff = np.minimum(np.abs(all_ad - all_fd) / (np.abs(all_fd) + 1e-8), 100.0)
    avg_rel_diff = np.mean(rel_diff)
    max_rel_diff = np.max(rel_diff)

    print(f"\n=== Gradient Timing Summary ===")
    print(f"  Samples: {num_samples}, Total time: {total_computation_time:.2f} s")
    print(f"  AD  total: {total_ad_time:.4f} s | mean: {avg_ad_time:.6f} s")
    print(f"  FD  total: {total_fd_time:.4f} s | mean: {avg_fd_time:.6f} s")
    print(f"  FD/AD speedup ratio: {speedup_ratio:.2f}x")
    print(f"  Mean relative error: {avg_rel_diff:.6f}")
    print(f"  Max  relative error: {max_rel_diff:.6f}")

    timing_results = {
        'num_samples': num_samples,
        'total_computation_time': total_computation_time,
        'total_ad_time': total_ad_time,
        'total_fd_time': total_fd_time,
        'avg_ad_time': avg_ad_time,
        'avg_fd_time': avg_fd_time,
        'std_ad_time': std_ad_time,
        'std_fd_time': std_fd_time,
        'speedup_ratio': speedup_ratio,
        'ad_times': ad_times,
        'fd_times': fd_times,
    }

    grad_diff = all_ad - all_fd
    pod_modes = np.tile(np.arange(1, max_modes + 1), num_samples)
    df = pd.DataFrame({
        "POD Mode": pod_modes.astype(str),
        "Gradient Difference (AD - FD)": grad_diff.flatten(),
    })

    plt.rcParams["axes.unicode_minus"] = False

    grad_base = save_path.replace('.pdf', '_grad.pdf') if save_path else \
        "./results/gradient_distribution.pdf"
    plt.figure(figsize=(16, 6))
    ax1 = plt.gca()
    df_sub = df[df["POD Mode"].isin([str(i) for i in range(1, 11)])]
    sns.violinplot(x="POD Mode", y="Gradient Difference (AD - FD)",
                   data=df_sub, inner=None, palette="Set2", ax=ax1)
    for pc in ax1.collections:
        pc.set_alpha(0.6)
    sns.boxplot(x="POD Mode", y="Gradient Difference (AD - FD)", data=df_sub,
                width=0.2, showcaps=True,
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1},
                showfliers=False,
                whiskerprops={'linewidth': 1},
                medianprops={'linewidth': 2, 'color': 'firebrick'},
                ax=ax1)
    sns.stripplot(x="POD Mode", y="Gradient Difference (AD - FD)", data=df_sub,
                  color="black", size=3, jitter=True, dodge=True, alpha=0.3, ax=ax1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                label=r'Perfect Agreement ($\nabla f_{\mathrm{AD}} = \nabla f_{\mathrm{FD}}$)')
    ax1.set_xlabel("Modes", fontsize=LABEL_SIZE + 2, fontweight="bold")
    ax1.set_ylabel(r"$\nabla f_{\mathrm{AD}} - \nabla f_{\mathrm{FD}}$",
                   fontsize=LABEL_SIZE + 2, fontweight="bold")
    ax1.set_yscale('symlog', linthresh=1e-3)
    ax1.tick_params(axis="both", labelsize=LABEL_SIZE)
    ax1.legend(fontsize=LABEL_SIZE, frameon=False)
    plt.tight_layout()
    plt.savefig(grad_base, format="pdf", dpi=650, bbox_inches='tight')
    plt.savefig(grad_base.replace('.pdf', '.jpg'), dpi=650, bbox_inches='tight')
    plt.close()

    time_base = save_path.replace('.pdf', '_time.pdf') if save_path else \
        "./results/timing_comparison.pdf"
    plt.figure(figsize=(4, 6))
    ax2 = plt.gca()
    methods = ['AD', 'FD']
    t_vals = [total_ad_time, total_fd_time]
    colors = ['teal', 'orange']
    bars = ax2.bar(methods, t_vals, color=colors, alpha=0.3,
                   edgecolor='black', linewidth=1)
    for bar, tv in zip(bars, t_vals):
        h_val = bar.get_height()
        lbl = (f"{tv:.3f}s" if tv < 1 else f"{tv:.1f}s")
        ax2.text(bar.get_x() + bar.get_width() / 2., h_val * 1.01,
                 lbl, ha='center', va='bottom',
                 fontsize=LABEL_SIZE + 2, fontweight='bold')
    ax2.set_ylabel("Total computation time (s)", fontsize=LABEL_SIZE + 2,
                   fontweight="bold")
    ax2.tick_params(axis="both", labelsize=LABEL_SIZE)
    ax2.set_yscale('log')
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.1)
    plt.tight_layout()
    plt.savefig(time_base, format="pdf", dpi=650, bbox_inches='tight')
    plt.savefig(time_base.replace('.pdf', '.jpg'), dpi=650, bbox_inches='tight')
    plt.close()

    return avg_rel_diff, max_rel_diff, timing_results


def visualize_gradients(gradients, max_modes=10, save_path=None):
   
    if isinstance(gradients, torch.Tensor):
        if gradients.is_cuda:
            gradients = gradients.cpu()
        g_flat = gradients.detach().numpy().reshape(-1)[:max_modes]
    else:
        g_flat = np.array(gradients).reshape(-1)[:max_modes]

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, max_modes + 1), g_flat)
    plt.xlabel('POD Mode', fontsize=12)
    plt.ylabel('Gradient', fontsize=12)
    plt.title(f'QoI Gradient w.r.t. First {max_modes} POD Modes', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if save_path is None:
        save_path = './results/pod_gradients.jpg'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gradient bar chart saved to '{save_path}'")


def analyze_multiple_samples(model, pod_coeffs_samples, device, h=1e-7,
                              num_samples=5, save_dir=None):
  
    num_samples = min(num_samples, len(pod_coeffs_samples))
    avg_errors, max_errors = [], []

    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}")
        sp = pod_coeffs_samples[i: i + 1]
        path = None
        if save_dir:
            path = os.path.join(save_dir, f'gradient_comparison_sample_{i+1}.jpg')
        ae, me = compare_gradients(model, sp, device=device, h=h,
                                   max_modes=10, save_path=path)
        avg_errors.append(ae)
        max_errors.append(me)

    print("\nMulti-sample gradient analysis:")
    print(f"  Mean relative error: {np.mean(avg_errors):.6f} ± {np.std(avg_errors):.6f}")
    print(f"  Max  relative error: {np.mean(max_errors):.6f} ± {np.std(max_errors):.6f}")
    return avg_errors, max_errors


def test_step_sizes(model, pod_coeffs, device,
                    steps=None, save_path=None):
  
    if steps is None:
        steps = [1e-3, 1e-5, 1e-7, 1e-9]

    ad_ref = compute_gradients(model, pod_coeffs, device=device)
    ad_ref = ad_ref.cpu().detach().numpy().reshape(-1)[:10]

    avg_errors, max_errors = [], []
    plt.figure(figsize=(15, 10))

    for i, step in enumerate(steps):
        print(f"\nTesting h={step:.1e}")
        fd = compute_finite_diff_gradients(model, pod_coeffs, device, h=step)
        fd_flat = fd.reshape(-1)[:10]

        rel = np.abs(ad_ref - fd_flat) / (np.abs(fd_flat) + 1e-10)
        avg_errors.append(np.mean(rel))
        max_errors.append(np.max(rel))
        print(f"  Mean rel. error: {avg_errors[-1]:.6f}")
        print(f"  Max  rel. error: {max_errors[-1]:.6f}")

        x = np.arange(1, 11)
        width = 0.35
        plt.subplot(len(steps), 1, i + 1)
        plt.bar(x - width / 2, ad_ref, width, label='AD')
        plt.bar(x + width / 2, fd_flat, width, label=f'FD (h={step:.1e})')
        plt.ylabel('Gradient', fontsize=12)
        plt.title(
            f'h={step:.1e} | Avg={avg_errors[-1]:.4f}, Max={max_errors[-1]:.4f}',
            fontsize=14,
        )
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel('POD Mode', fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = './results/step_size_comparison.jpg'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Step-size study saved to '{save_path}'")

    return avg_errors, max_errors
