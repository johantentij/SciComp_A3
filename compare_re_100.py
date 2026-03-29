"""
run_challenge_a.py — Unified entry point for Challenge A.

Edit the parameters below, then run:
    export PYTHONPATH="/usr/local/lib/python3.12/site-packages:$PYTHONPATH"
    python run_challenge_a.py

Each solver will:
  1. Run the simulation
  2. Save a vorticity GIF
  3. Save final vorticity + velocity PNGs
  4. Print Strouhal number and timing info
"""

import sys
if "/usr/local/lib/python3.12/site-packages" not in sys.path:
    sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

import numpy as np
import time as time_mod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from geometry import CYL_X, CYL_Y, CYL_R, L_X, L_Y

# ============================================================
# >>>  EDIT PARAMETERS HERE  <<<
# ============================================================

Re = 100                 # Reynolds number

# --- LBM ---
LBM_ENABLED   = True
LBM_Ny        = 80       # lattice nodes in y (higher = finer)
LBM_U_max     = 0.05     # max lattice velocity (keep < 0.1)
LBM_steps     = 20000    # total time steps
LBM_frame_every = 200    # capture frame every N steps

# --- FD ---
FD_ENABLED    = True
FD_Nx         = 800       # grid cells in x
FD_p_iters    = 50        # pressure Poisson iterations per step
FD_steps      = 20000
FD_frame_every = 500

# --- FEM ---
FEM_ENABLED   = True
FEM_max_h     = 0.03      # max mesh element size
FEM_dt        = 0.001     # time step [s]
FEM_steps     = 20000
FEM_frame_every = 100
FEM_Nx_interp = 200       # grid resolution for interpolation

# ============================================================
# >>>  END OF PARAMETERS  <<<
# ============================================================


def _draw_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(CYL_X + CYL_R * np.cos(theta),
            CYL_Y + CYL_R * np.sin(theta),
            color='gray', zorder=5)


def _save_gif(frames, X, Y, solid_mask, method, Re, dt_per_frame,
              filename):
    """Build and save a vorticity GIF from a list of frames."""
    # Color scale from all frames
    all_vals = np.concatenate([np.abs(f[~solid_mask]).ravel()
                               for f in frames])
    all_vals = all_vals[all_vals > 0]
    vmax = np.percentile(all_vals, 95) if len(all_vals) > 0 else 0.01
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.pcolormesh(X, Y, frames[0], cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    _draw_cylinder(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    title = ax.set_title('')
    plt.colorbar(im, ax=ax, label='Vorticity', shrink=0.8)
    plt.tight_layout()

    def update(i):
        im.set_array(frames[i].ravel())
        if dt_per_frame is not None:
            title.set_text(f'{method} Vorticity — Re={Re}, '
                           f't={dt_per_frame*(i+1):.3f}s')
        else:
            title.set_text(f'{method} Vorticity — Re={Re}, '
                           f'frame {i+1}/{len(frames)}')
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=80, blit=False)
    anim.save(filename, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"  Saved {filename}")


def _save_snapshot(X, Y, vort, speed, solid_mask, method, Re, prefix):
    """Save final vorticity and velocity PNGs."""
    fluid_vort = np.abs(vort[~solid_mask])
    fluid_vort = fluid_vort[fluid_vort > 0]
    vmax = np.percentile(fluid_vort, 95) if len(fluid_vort) > 0 else 0.01

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.pcolormesh(X, Y, vort, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  shading='auto')
    _draw_cylinder(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{method} Vorticity — Re={Re}')
    plt.tight_layout()
    plt.savefig(f'{prefix}_vorticity.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')
    _draw_cylinder(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{method} |u| — Re={Re}')
    plt.tight_layout()
    plt.savefig(f'{prefix}_velocity.png', dpi=150)
    plt.close(fig)

    print(f"  Saved {prefix}_vorticity.png, {prefix}_velocity.png")


# ============================================================
# LBM
# ============================================================
def run_lbm():
    from solver_lbm import LBMSolver

    print(f"\n{'='*60}")
    print(f"  LBM — Re={Re}, Ny={LBM_Ny}, steps={LBM_steps}")
    print(f"{'='*60}")

    solver = LBMSolver(Re=Re, Ny=LBM_Ny, U_max_lbm=LBM_U_max)
    X, Y = solver.get_coords()

    frames = []
    t0 = time_mod.time()

    for t in range(LBM_steps):
        solver.step()
        if (t + 1) % LBM_frame_every == 0:
            frames.append(solver.get_vorticity().copy())
        if (t + 1) % 5000 == 0:
            max_u = np.sqrt(solver.ux**2 + solver.uy**2).max()
            print(f"  Step {t+1}/{LBM_steps}  max|u|={max_u:.5f}")
            if np.isnan(max_u) or max_u > 0.4:
                print("  UNSTABLE — stopping")
                break

    wall_time = time_mod.time() - t0
    steps_done = min(t + 1, LBM_steps)
    sps = steps_done / wall_time

    St = solver.get_strouhal()
    print(f"  Time: {wall_time:.1f}s ({sps:.0f} steps/s)")
    print(f"  Strouhal: {St:.3f}" if St else "  Strouhal: not detected")

    # GIF
    _save_gif(frames, X, Y, solver.solid, 'LBM', Re,
              dt_per_frame=None, filename='lbm_animation.gif')

    # Snapshots
    speed = np.sqrt(solver.ux**2 + solver.uy**2)
    _save_snapshot(X, Y, solver.get_vorticity(), speed,
                   solver.solid, 'LBM', Re, 'lbm')

    return {'St': St, 'steps_per_sec': sps, 'wall_time': wall_time}


# ============================================================
# FD
# ============================================================
def run_fd():
    from solver_fd import FDSolver

    print(f"\n{'='*60}")
    print(f"  FD — Re={Re}, Nx={FD_Nx}, steps={FD_steps}")
    print(f"{'='*60}")

    solver = FDSolver(Re=Re, Nx=FD_Nx, p_iters=FD_p_iters)
    X, Y = solver.get_coords()

    frames = []
    t0 = time_mod.time()

    # JIT warmup
    print("  Compiling JIT...")
    solver.step()

    for t in range(1, FD_steps):
        solver.step()
        if (t + 1) % FD_frame_every == 0:
            frames.append(solver.get_vorticity().copy())
        if (t + 1) % 5000 == 0:
            max_u = np.sqrt(solver.u**2 + solver.v**2).max()
            print(f"  Step {t+1}/{FD_steps}  max|u|={max_u:.4f}")
            if np.isnan(max_u) or max_u > 10:
                print("  DIVERGED — stopping")
                break

    wall_time = time_mod.time() - t0
    steps_done = min(t + 1, FD_steps)
    sps = steps_done / wall_time

    St = solver.get_strouhal()
    print(f"  Time: {wall_time:.1f}s ({sps:.0f} steps/s)")
    print(f"  Strouhal: {St:.3f}" if St else "  Strouhal: not detected")

    # GIF
    dt_frame = FD_frame_every * solver.dt
    _save_gif(frames, X, Y, solver.solid, 'FD', Re,
              dt_per_frame=dt_frame, filename='fd_animation.gif')

    # Snapshots
    speed = np.sqrt(solver.u**2 + solver.v**2)
    _save_snapshot(X, Y, solver.get_vorticity(), speed,
                   solver.solid, 'FD', Re, 'fd')

    return {'St': St, 'steps_per_sec': sps, 'wall_time': wall_time}


# ============================================================
# FEM
# ============================================================
def run_fem():
    from solver_fem import FEMSolver

    print(f"\n{'='*60}")
    print(f"  FEM — Re={Re}, max_h={FEM_max_h}, dt={FEM_dt}, "
          f"steps={FEM_steps}")
    print(f"{'='*60}")

    solver = FEMSolver(Re=Re, max_h=FEM_max_h, dt=FEM_dt)

    frames = []
    frames_X, frames_Y = None, None
    t0 = time_mod.time()

    for t in range(FEM_steps):
        solver.step()
        if (t + 1) % FEM_frame_every == 0:
            X, Y, vort = solver.get_vorticity(Nx=FEM_Nx_interp)
            frames.append(vort.copy())
            if frames_X is None:
                frames_X, frames_Y = X, Y
        if (t + 1) % 1000 == 0:
            print(f"  Step {t+1}/{FEM_steps} (t={solver.time:.3f}s)")

    wall_time = time_mod.time() - t0
    sps = FEM_steps / wall_time

    St = solver.get_strouhal()
    print(f"  Time: {wall_time:.1f}s ({sps:.1f} steps/s)")
    print(f"  Strouhal: {St:.3f}" if St else "  Strouhal: not detected")

    # GIF
    # Build a simple solid mask for plotting (cylinder region on interp grid)
    solid_interp = ((frames_X - CYL_X)**2 + (frames_Y - CYL_Y)**2
                    <= CYL_R**2)
    dt_frame = FEM_frame_every * FEM_dt
    _save_gif(frames, frames_X, frames_Y, solid_interp, 'FEM', Re,
              dt_per_frame=dt_frame, filename='fem_animation.gif')

    # Final snapshot
    X, Y, ux, uy = solver.get_numpy_fields(Nx=FEM_Nx_interp)
    speed = np.sqrt(ux**2 + uy**2)
    _, _, vort = solver.get_vorticity(Nx=FEM_Nx_interp)
    _save_snapshot(X, Y, vort, speed, solid_interp, 'FEM', Re, 'fem')

    return {'St': St, 'steps_per_sec': sps, 'wall_time': wall_time}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    results = {}

    if LBM_ENABLED:
        results['LBM'] = run_lbm()

    if FD_ENABLED:
        results['FD'] = run_fd()

    if FEM_ENABLED:
        results['FEM'] = run_fem()

    # Summary table
    print(f"\n{'='*65}")
    print(f"  SUMMARY — Re={Re}")
    print(f"{'='*65}")
    print(f"{'Method':<8} {'St':<10} {'Steps/s':<12} {'Wall time':<12}")
    print(f"{'-'*65}")
    for name, info in results.items():
        st = f"{info['St']:.3f}" if info['St'] else "—"
        print(f"{name:<8} {st:<10} {info['steps_per_sec']:<12.0f} "
              f"{info['wall_time']:<12.1f}s")
    print(f"{'='*65}")
