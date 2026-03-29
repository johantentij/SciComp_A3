"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver
Numba Optimized + Smagorinsky LES Subgrid Model
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# =============================================================================
# 1.  D2Q9 Lattice Definition
# =============================================================================

c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
              [1, 1], [-1, 1], [-1, -1], [1, -1]])

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])    

opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# =============================================================================
# 2.  Simulation Parameters
# =============================================================================

Nx = 300
Ny = 120

cx_cyl = Nx // 5
cy_cyl = Ny // 2
r_cyl  = 8

U_inlet = 0.12
Re      = 2000      # <--- Bumped to 2000! (Would crash instantly without LES)
C_s     = 0.16      # <--- Smagorinsky constant (typically 0.1 to 0.2)

D   = 2 * r_cyl
nu  = U_inlet * D / Re
tau = 3.0 * nu + 0.5

print(f"Simulation parameters:")
print(f"  Grid:      {Nx} x {Ny}")
print(f"  Re={Re},  U_inlet={U_inlet},  nu={nu:.6f},  tau_0={tau:.4f}")
print(f"  Smagorinsky Constant C_s = {C_s}")

# =============================================================================
# 3.  Obstacle Mask 
# =============================================================================

x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
obstacle = (X - cx_cyl)**2 + (Y - cy_cyl)**2 <= r_cyl**2

# =============================================================================
# 4.  Numba-Optimized Core Functions
# =============================================================================

@njit
def equilibrium(rho, ux, uy, c, w):
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9))
    for x in range(Nx):
        for y in range(Ny):
            usqr = ux[x, y]**2 + uy[x, y]**2
            for i in range(9):
                cu = c[i, 0] * ux[x, y] + c[i, 1] * uy[x, y]
                feq[x, y, i] = w[i] * rho[x, y] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)
    return feq

@njit
def lbm_step(f, f_out, obstacle, U_inlet, tau_0, C_s, c, w, opp):
    Nx, Ny, _ = f.shape
    rho = np.zeros((Nx, Ny))
    ux  = np.zeros((Nx, Ny))
    uy  = np.zeros((Nx, Ny))

    # --- 1. Macroscopic, LES Collision, and Bounce-back ---
    for x in range(Nx):
        for y in range(Ny):
            if obstacle[x, y]:
                for i in range(9):
                    f_out[x, y, i] = f[x, y, opp[i]]
                rho[x, y] = 1.0
                ux[x, y] = 0.0
                uy[x, y] = 0.0
            else:
                r = 0.0
                u_x = 0.0
                u_y = 0.0
                for i in range(9):
                    r += f[x, y, i]
                    u_x += f[x, y, i] * c[i, 0]
                    u_y += f[x, y, i] * c[i, 1]
                
                u_x /= r
                u_y /= r
                rho[x, y] = r
                ux[x, y] = u_x
                uy[x, y] = u_y
                
                # Pre-calculate local equilibrium
                usqr = u_x**2 + u_y**2
                feq = np.empty(9)
                for i in range(9):
                    cu = c[i, 0] * u_x + c[i, 1] * u_y
                    feq[i] = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)
                
                # --- SMAGORINSKY LES ADDITION ---
                Qxx = 0.0
                Qyy = 0.0
                Qxy = 0.0
                for i in range(9):
                    neq = f[x, y, i] - feq[i]
                    Qxx += c[i, 0] * c[i, 0] * neq
                    Qyy += c[i, 1] * c[i, 1] * neq
                    Qxy += c[i, 0] * c[i, 1] * neq
                
                # Stress tensor magnitude
                Q_mag = np.sqrt(Qxx**2 + Qyy**2 + 2.0 * Qxy**2)
                
                # Effective relaxation time via quadratic solution
                tau_eff = 0.5 * (tau_0 + np.sqrt(tau_0**2 + 18.0 * (C_s**2) * Q_mag / r))
                # --------------------------------
                
                # BGK Collision using tau_eff
                for i in range(9):
                    f_out[x, y, i] = f[x, y, i] - (f[x, y, i] - feq[i]) / tau_eff

    # --- 2. Streaming ---
    for x in range(Nx):
        for y in range(Ny):
            for i in range(9):
                src_x = (x - c[i, 0]) % Nx
                src_y = (y - c[i, 1]) % Ny
                f[x, y, i] = f_out[src_x, src_y, i]

    # --- 3. Boundary Conditions ---
    for y in range(Ny):
        for i in range(9):
            f[Nx-1, y, i] = f[Nx-2, y, i]

    for y in range(Ny):
        if not obstacle[0, y]:
            rho_in = (f[0, y, 0] + f[0, y, 2] + f[0, y, 4] +
                      2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])) / (1.0 - U_inlet)
            f[0, y, 1] = f[0, y, 3] + (2.0/3.0) * rho_in * U_inlet
            f[0, y, 5] = f[0, y, 7] - 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0/6.0) * rho_in * U_inlet
            f[0, y, 8] = f[0, y, 6] + 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0/6.0) * rho_in * U_inlet

    return rho, ux, uy

# =============================================================================
# 5.  Main Function
# =============================================================================

def main():
    rho_init = np.ones((Nx, Ny))
    ux_init  = np.full((Nx, Ny), U_inlet)
    uy_init  = np.zeros((Nx, Ny))

    uy_init += 0.001 * U_inlet * np.sin(2.0 * np.pi * Y / Ny)
    ux_init[obstacle] = 0.0
    uy_init[obstacle] = 0.0

    f = equilibrium(rho_init, ux_init, uy_init, c, w)
    f_out = np.zeros_like(f)

    plot_mode = 'velocity'

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    def plot_velocity(ux, uy, step):
        speed = np.sqrt(ux**2 + uy**2)
        speed[obstacle] = np.nan
        ax.clear()
        ax.imshow(speed.T, origin='lower', cmap='jet', 
                  vmin=0, vmax=U_inlet * 2.0, aspect='auto', extent=[0, Nx, 0, Ny])
        ax.set_title(f"Velocity magnitude — step {step}")
        fig.tight_layout()
        plt.pause(0.01)

    def plot_vorticity(ux, uy, step):
        vorticity = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)
                   - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1))
        vorticity[obstacle] = np.nan
        ax.clear()
        ax.imshow(vorticity.T, origin='lower', cmap='RdBu_r',
                  vmin=-0.08, vmax=0.08, aspect='auto', extent=[0, Nx, 0, Ny])
        ax.set_title(f"Vorticity field (LES Re={Re}) — step {step}")
        fig.tight_layout()
        plt.pause(0.01)

    def plot_field(ux, uy, step):
        if plot_mode == 'vorticity':
            plot_vorticity(ux, uy, step)
        elif plot_mode == 'velocity':
            plot_velocity(ux, uy, step)

    n_steps  = 30000
    plot_every = 100 

    print(f"\nRunning {n_steps} timesteps ...")

    for step in range(1, n_steps + 1):
        # Pass C_s into our Numba function
        rho, ux, uy = lbm_step(f, f_out, obstacle, U_inlet, tau, C_s, c, w, opp)

        if step % plot_every == 0:
            plot_field(ux, uy, step)

        if step % 1000 == 0:
            avg_rho = np.mean(rho[~obstacle])
            print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

    print("\nSimulation complete.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()