"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver
=============================================================
A minimal, educational 2D LBM solver using the D2Q9 lattice and BGK collision
operator.  This code is prepared for the student of the course Scientific Computing at UvA.
The code simulates flow past a circular cylinder at Reynolds number 150 to produce the
classic Karman vortex street.

Note: Not the exact implementation of the benchmark case from the original paper, 
but a simplified version that captures the essential physics and flow features.  
The code is structured for clarity and educational purposes, not for maximum 
performance or accuracy.

Algorithm overview (each timestep):
  1. Compute macroscopic quantities (density, velocity) from distributions
  2. Collision step  — relax f toward local equilibrium (BGK)
  3. Bounce-back    — reflect populations at obstacle nodes
  4. Streaming step — propagate f_i along lattice velocity c_i
  5. Boundary conditions — Zou-He inlet, open outlet

Dependencies: numpy, matplotlib
Usage:        python lbm_karman.py
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1.  D2Q9 Lattice Definition
# =============================================================================
#
#   6  2  5        Lattice velocities c_i (i = 0..8):
#    \ | /           0: rest          (0, 0)
#   3--0--1          1-4: axis-aligned  (±1,0), (0,±1)
#    / | \           5-8: diagonals     (±1,±1)
#   7  4  8
#
#  Each row of `c` is a velocity vector [cx, cy] for direction i.

c = np.array([[0, 0],    # 0  — rest
              [1, 0],    # 1  — east
              [0, 1],    # 2  — north
              [-1, 0],   # 3  — west
              [0, -1],   # 4  — south
              [1, 1],    # 5  — north-east
              [-1, 1],   # 6  — north-west
              [-1, -1],  # 7  — south-west
              [1, -1]])  # 8  — south-east

# Lattice weights (from the D2Q9 equilibrium derivation)
w = np.array([4/9,                        # rest
              1/9, 1/9, 1/9, 1/9,         # axis-aligned
              1/36, 1/36, 1/36, 1/36])    # diagonals

# Opposite direction index for each i (used in bounce-back)
# e.g. opposite of 1 (east) is 3 (west)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# =============================================================================
# 2.  Simulation Parameters
# =============================================================================

Nx = 300        # domain length  (lattice units)
Ny = 120        # domain height  (lattice units)

# Cylinder geometry
cx_cyl = Nx // 5    # cylinder center x  (1/5 from inlet)
cy_cyl = Ny // 2    # cylinder center y  (centered vertically)
r_cyl  = 8          # cylinder radius

# Flow parameters
U_inlet = 0.12      # inlet velocity (lattice units, keep ≪ 1 for low Mach)
Re      = 150       # target Reynolds number

# Derived quantities:
#   Re = U * D / nu   →  nu = U * D / Re
#   In LBM:  nu = cs² * (tau - 0.5)  where cs² = 1/3
#   Therefore:  tau = 3 * nu + 0.5
D   = 2 * r_cyl                             # cylinder diameter
nu  = U_inlet * D / Re                      # kinematic viscosity
tau = 3.0 * nu + 0.5                        # BGK relaxation time

print(f"Simulation parameters:")
print(f"  Grid:      {Nx} x {Ny}")
print(f"  Cylinder:  center=({cx_cyl},{cy_cyl}), r={r_cyl}, D={D}")
print(f"  Re={Re},  U_inlet={U_inlet},  nu={nu:.6f},  tau={tau:.4f}")

# =============================================================================
# 3.  Equilibrium Distribution Function
# =============================================================================

def equilibrium(rho, ux, uy):
    """
    Compute the equilibrium distribution f^eq for the D2Q9 lattice.

    The equilibrium is derived from a second-order Taylor expansion of the
    Maxwell-Boltzmann distribution:

        f_i^eq = w_i * rho * (1 + c_i·u/cs² + (c_i·u)²/(2·cs⁴) - u·u/(2·cs²))

    where cs² = 1/3  (lattice speed of sound squared).

    Parameters
    ----------
    rho : ndarray (Nx, Ny)   — macroscopic density
    ux  : ndarray (Nx, Ny)   — x-component of velocity
    uy  : ndarray (Nx, Ny)   — y-component of velocity

    Returns
    -------
    feq : ndarray (Nx, Ny, 9) — equilibrium distributions
    """
    feq = np.zeros((Nx, Ny, 9))
    usqr = ux**2 + uy**2                 # |u|²

    for i in range(9):
        cu = c[i, 0] * ux + c[i, 1] * uy  # c_i · u
        feq[:, :, i] = w[i] * rho * (1.0
                                      + 3.0 * cu            # c_i·u / cs²
                                      + 4.5 * cu**2         # (c_i·u)² / (2·cs⁴)
                                      - 1.5 * usqr)         # -|u|² / (2·cs²)
    return feq
    
# =============================================================================
# 4.  Obstacle Mask  (circular cylinder)
# =============================================================================

# Boolean array: True where the obstacle is located
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y, indexing='ij')    # X,Y have shape (Nx, Ny)
obstacle = (X - cx_cyl)**2 + (Y - cy_cyl)**2 <= r_cyl**2

# =============================================================================
# 5.  Main Function
# =============================================================================

def main():
    """Run the LBM simulation: initialization, time loop, and visualization."""

    # =================================================================
    # 5a. Initialization
    # =================================================================

    # Start with uniform flow at inlet velocity everywhere
    rho_init = np.ones((Nx, Ny))
    ux_init  = np.full((Nx, Ny), U_inlet)
    uy_init  = np.zeros((Nx, Ny))

    # Small transverse perturbation to break symmetry and trigger vortex shedding
    uy_init += 0.001 * U_inlet * np.sin(2.0 * np.pi * Y / Ny)

    # Set velocity to zero inside the obstacle
    ux_init[obstacle] = 0.0
    uy_init[obstacle] = 0.0

    # Initialize distributions to equilibrium
    f = equilibrium(rho_init, ux_init, uy_init)

    # =================================================================
    # 6.  Visualization Setup
    # =================================================================

    # Visualization mode: 'velocity' (default) or 'vorticity' or 'none'
    plot_mode = 'velocity'

    plt.ion()                        # interactive mode for live animation
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    def plot_velocity(ux, uy, step):
        """Plot the velocity magnitude field |u| = sqrt(ux² + uy²)."""
        speed = np.sqrt(ux**2 + uy**2)
        speed[obstacle] = np.nan      # mask cylinder

        ax.clear()
        ax.imshow(speed.T, origin='lower', cmap='jet', # Warning!: Not a sequential color scale, use 'viridis' instead!
                  vmin=0, vmax=U_inlet * 2.0, aspect='auto',
                  extent=[0, Nx, 0, Ny])
        ax.set_title(f"Velocity magnitude — step {step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        plt.pause(0.01)

    def plot_vorticity(ux, uy, step):
        """
        Plot the vorticity field (curl of velocity).
        Vorticity = ∂uy/∂x - ∂ux/∂y  — highlights the alternating vortices
        in the Karman street much more vividly than velocity magnitude.
        """
        vorticity = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)
                   - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1))
        vorticity[obstacle] = np.nan

        ax.clear()
        ax.imshow(vorticity.T, origin='lower', cmap='RdBu_r',
                  vmin=-0.04, vmax=0.04, aspect='auto',
                  extent=[0, Nx, 0, Ny])
        ax.set_title(f"Vorticity field — step {step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        plt.pause(0.01)

    def plot_field(ux, uy, step):
        """Dispatch to the selected visualization mode."""
        if plot_mode == 'vorticity':
            plot_vorticity(ux, uy, step)
        elif plot_mode == 'velocity':
            plot_velocity(ux, uy, step)
        else:
            pass  # no plotting

    # =================================================================
    # 7.  Main Simulation Loop
    # =================================================================
    #
    #  The LBM algorithm follows the "collide-then-stream" pattern:
    #    1. Compute macroscopic quantities (rho, u) from current distributions
    #    2. Collision:  relax f toward equilibrium  →  f_out
    #    3. Bounce-back: at obstacle nodes, replace f_out with reflected f
    #    4. Streaming:  propagate f_out along lattice velocities  →  f
    #    5. Boundary conditions (inlet/outlet) applied to post-streaming f
    #
    #  np.roll provides periodic wrapping, which serves as the y-boundary
    #  condition (periodic in the vertical direction).

    n_steps  = 30000    # total number of timesteps
    plot_every = 25     # plot interval (steps)

    print(f"\nRunning {n_steps} timesteps ...")

    for step in range(1, n_steps + 1):

        # -------------------------------------------------------------
        # 7a.  Macroscopic quantities: density and velocity
        #      rho = Σ f_i,   rho·u = Σ c_i · f_i
        # -------------------------------------------------------------
        rho = np.sum(f, axis=2)
        ux  = np.sum(f * c[:, 0], axis=2) / rho
        uy  = np.sum(f * c[:, 1], axis=2) / rho

        # -------------------------------------------------------------
        # 7b.  Collision step (BGK single-relaxation-time)
        #      f_out_i = f_i - (f_i - f_i^eq) / tau
        #
        #      f_out is a NEW array so that f (pre-collision) is
        #      preserved for the bounce-back step below.
        # -------------------------------------------------------------
        feq   = equilibrium(rho, ux, uy)
        f_out = f - (f - feq) / tau

        # -------------------------------------------------------------
        # 7c.  Bounce-back on obstacle (no-slip wall condition)
        #      At obstacle nodes, replace post-collision populations
        #      with PRE-collision populations from the opposite
        #      direction.  When streamed, they travel back the way
        #      they came — reflecting off the obstacle surface.
        # -------------------------------------------------------------
        for i in range(9):
            f_out[obstacle, i] = f[obstacle, opp[i]]

        # -------------------------------------------------------------
        # 7d.  Streaming step
        #      Shift each f_i by its lattice velocity c_i.
        #      np.roll provides periodic wrapping in y.
        # -------------------------------------------------------------
        for i in range(9):
            f[:, :, i] = np.roll(f_out[:, :, i], shift=c[i, 0], axis=0)
            f[:, :, i] = np.roll(f[:, :, i],     shift=c[i, 1], axis=1)

        # -------------------------------------------------------------
        # 7e.  Outlet boundary condition (zero-gradient / open)
        #      Copy from second-to-last column so vortices can leave.
        # -------------------------------------------------------------
        f[-1, :, :] = f[-2, :, :]

        # -------------------------------------------------------------
        # 7f.  Inlet boundary condition (Zou-He, fixed velocity)
        #      After streaming, populations 1, 5, 8 at x=0 are unknown
        #      (they would come from outside the domain).  Zou-He
        #      determines them from known populations and prescribed
        #      inlet velocity (ux=U_inlet, uy=0).
        # -------------------------------------------------------------
        rho_in = (  (f[0, :, 0] + f[0, :, 2] + f[0, :, 4])
                  + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
                 ) / (1.0 - U_inlet)

        f[0, :, 1] = f[0, :, 3] + (2.0/3.0) * rho_in * U_inlet
        f[0, :, 5] = (f[0, :, 7]
                      - 0.5 * (f[0, :, 2] - f[0, :, 4])
                      + (1.0/6.0) * rho_in * U_inlet)
        f[0, :, 8] = (f[0, :, 6]
                      + 0.5 * (f[0, :, 2] - f[0, :, 4])
                      + (1.0/6.0) * rho_in * U_inlet)

        # -------------------------------------------------------------
        # 7g.  Visualization & progress
        # -------------------------------------------------------------
        if step % plot_every == 0:
            plot_field(ux, uy, step)

        if step % 1000 == 0:
            avg_rho = np.mean(rho[~obstacle])
            print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

    # =================================================================
    # 8.  Final Output
    # =================================================================

    print("\nSimulation complete.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
    
