"""
visualization.py — Shared plotting utilities for comparing FD, LBM, FEM results.

Provides functions for:
  - Side-by-side vorticity comparison
  - Probe signal overlay
  - Strouhal number comparison
  - Performance summary table
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import CYL_X, CYL_Y, CYL_R


def _draw_cylinder(ax):
    """Draw the cylinder on an axis."""
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(CYL_X + CYL_R * np.cos(theta),
            CYL_Y + CYL_R * np.sin(theta),
            color='gray', zorder=5)


def plot_vorticity(X, Y, vort, title='Vorticity', filename=None,
                   solid_mask=None):
    """Plot a single vorticity field."""
    fig, ax = plt.subplots(figsize=(15, 3.5))

    if solid_mask is not None:
        vort_fluid = np.abs(vort[~solid_mask])
    else:
        vort_fluid = np.abs(vort[vort != 0])

    vmax = np.percentile(vort_fluid, 95) if len(vort_fluid) > 0 else 1.0
    vmax = max(vmax, 0.01)

    im = ax.pcolormesh(X, Y, vort, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    _draw_cylinder(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Vorticity', shrink=0.8)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved {filename}")
    return fig


def plot_comparison_vorticity(results, filename='comparison_vorticity.png'):
    """
    Plot vorticity fields from multiple methods side by side.

    Parameters
    ----------
    results : dict
        Keys are method names (e.g. 'FD', 'LBM', 'FEM'),
        values are dicts with keys 'X', 'Y', 'vort', optionally 'solid'.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.2 * n + 0.5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        vort = data['vort']
        solid = data.get('solid', None)

        if solid is not None:
            vort_vals = np.abs(vort[~solid])
        else:
            vort_vals = np.abs(vort[vort != 0])

        vmax = np.percentile(vort_vals, 95) if len(vort_vals) > 0 else 1.0
        vmax = max(vmax, 0.01)

        ax.pcolormesh(data['X'], data['Y'], vort,
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        _draw_cylinder(ax)
        ax.set_aspect('equal')
        ax.set_ylabel('y [m]')
        ax.set_title(f'{name}')

    axes[-1].set_xlabel('x [m]')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    return fig


def plot_probe_comparison(probes, filename='comparison_probe.png'):
    """
    Overlay probe signals from multiple methods.

    Parameters
    ----------
    probes : dict
        Keys are method names, values are dicts with keys
        'time' (array) and 'signal' (array).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for name, data in probes.items():
        ax.plot(data['time'], data['signal'], label=name, alpha=0.8)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('v_y at probe')
    ax.set_title('Vortex Shedding: Probe Signal Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    return fig


def plot_convergence(convergence_data, filename='comparison_convergence.png'):
    """
    Plot convergence of delta (max change) vs iteration/time for iterative methods.

    Parameters
    ----------
    convergence_data : dict
        Keys: method names. Values: dict with 'x' (steps/iters) and 'y' (delta).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in convergence_data.items():
        ax.semilogy(data['x'], data['y'], label=name)

    ax.set_xlabel('Step / Iteration')
    ax.set_ylabel('Residual / delta')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    return fig


def print_summary(methods_info):
    """
    Print a summary table comparing the three methods.

    Parameters
    ----------
    methods_info : dict
        Keys: method names. Values: dict with optional keys:
        'Re', 'resolution', 'dt', 'n_steps', 'wall_time',
        'steps_per_sec', 'strouhal', 'max_stable_Re'.
    """
    print("\n" + "=" * 75)
    print(f"{'Method':<10} {'Resolution':<15} {'St':<8} "
          f"{'Steps/s':<10} {'Wall time':<12} {'Max Re':<8}")
    print("-" * 75)

    for name, info in methods_info.items():
        res = info.get('resolution', '—')
        st = info.get('strouhal', None)
        st_str = f"{st:.3f}" if st else "—"
        sps = info.get('steps_per_sec', None)
        sps_str = f"{sps:.0f}" if sps else "—"
        wt = info.get('wall_time', None)
        wt_str = f"{wt:.1f}s" if wt else "—"
        max_re = info.get('max_stable_Re', None)
        re_str = f"{max_re}" if max_re else "—"

        print(f"{name:<10} {str(res):<15} {st_str:<8} "
              f"{sps_str:<10} {wt_str:<12} {re_str:<8}")

    print("=" * 75 + "\n")
