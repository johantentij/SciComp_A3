"""
solver_fd.py — Finite Difference solver for the Karman vortex street.

Method: Chorin's projection (fractional step):
  1. Advection-diffusion: compute intermediate velocity u* (ignoring pressure)
  2. Pressure Poisson: solve nabla^2 p = (rho/dt) * div(u*)
  3. Correction: u^{n+1} = u* - (dt/rho) * grad(p)

Spatial discretization:
  - Collocated grid (u, v, p all at cell centers)
  - Advection: 2nd-order upwind
  - Diffusion: 2nd-order central
  - Pressure Poisson: Jacobi iteration (or SOR)

Boundary conditions:
  - Inlet (left):   parabolic velocity, zero pressure gradient
  - Outlet (right):  zero velocity gradient, p = 0
  - Top/bottom:     no-slip walls
  - Cylinder:       no-slip (immersed boundary via mask)

Grid indexing: u[j, i] where j is y-index, i is x-index.
"""

import numpy as np
import time as time_mod
from numba import njit, prange
from geometry import (
    L_X, L_Y, CYL_X, CYL_Y, CYL_R, CYL_D,
    u_inlet_parabolic, mean_velocity,
)


@njit(cache=True)
def _build_cylinder_mask(Nx, Ny, dx, dy):
    """Build boolean mask for cells inside the cylinder."""
    mask = np.zeros((Ny, Nx), dtype=np.bool_)
    for j in range(Ny):
        for i in range(Nx):
            x = (i + 0.5) * dx  # cell center
            y = (j + 0.5) * dy
            if (x - CYL_X)**2 + (y - CYL_Y)**2 <= CYL_R**2:
                mask[j, i] = True
    return mask


@njit(cache=True, parallel=True)
def _advect_diffuse(u, v, u_new, v_new, Nx, Ny, dx, dy, dt, nu, solid):
    """
    Compute intermediate velocity (u*, v*) with advection + diffusion.
    Uses first-order upwind for advection, central for diffusion.
    """
    for j in prange(1, Ny - 1):
        for i in range(1, Nx - 1):
            if solid[j, i]:
                u_new[j, i] = 0.0
                v_new[j, i] = 0.0
                continue

            uc = u[j, i]
            vc = v[j, i]

            # --- Advection (upwind) ---
            # du/dx
            if uc > 0:
                dudx = (u[j, i] - u[j, i-1]) / dx
            else:
                dudx = (u[j, i+1] - u[j, i]) / dx
            # du/dy
            if vc > 0:
                dudy = (u[j, i] - u[j-1, i]) / dy
            else:
                dudy = (u[j+1, i] - u[j, i]) / dy

            # dv/dx
            if uc > 0:
                dvdx = (v[j, i] - v[j, i-1]) / dx
            else:
                dvdx = (v[j, i+1] - v[j, i]) / dx
            # dv/dy
            if vc > 0:
                dvdy = (v[j, i] - v[j-1, i]) / dy
            else:
                dvdy = (v[j+1, i] - v[j, i]) / dy

            advection_u = uc * dudx + vc * dudy
            advection_v = uc * dvdx + vc * dvdy

            # --- Diffusion (central) ---
            diff_u = nu * (
                (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2
                + (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2
            )
            diff_v = nu * (
                (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / dx**2
                + (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / dy**2
            )

            u_new[j, i] = u[j, i] + dt * (-advection_u + diff_u)
            v_new[j, i] = v[j, i] + dt * (-advection_v + diff_v)


@njit(cache=True, parallel=True)
def _pressure_poisson(p, rhs, Nx, Ny, dx, dy, solid, n_iters, omega):
    """
    Solve pressure Poisson equation using SOR iteration.
    nabla^2 p = rhs
    """
    dx2 = dx * dx
    dy2 = dy * dy
    coeff = 2.0 * (1.0/dx2 + 1.0/dy2)

    for _ in range(n_iters):
        for j in prange(1, Ny - 1):
            for i in range(1, Nx - 1):
                if solid[j, i]:
                    continue

                p_new = (
                    (p[j, i+1] + p[j, i-1]) / dx2
                    + (p[j+1, i] + p[j-1, i]) / dy2
                    - rhs[j, i]
                ) / coeff

                p[j, i] = (1.0 - omega) * p[j, i] + omega * p_new

        # Boundary conditions for pressure
        # Left (inlet): dp/dx = 0
        for j in range(Ny):
            p[j, 0] = p[j, 1]
        # Right (outlet): p = 0
        for j in range(Ny):
            p[j, Nx-1] = 0.0
        # Top/bottom: dp/dy = 0
        for i in range(Nx):
            p[0, i] = p[1, i]
            p[Ny-1, i] = p[Ny-2, i]

    return p


@njit(cache=True, parallel=True)
def _correct_velocity(u, v, p, Nx, Ny, dx, dy, dt, solid):
    """Correct velocity with pressure gradient: u = u* - dt * grad(p)."""
    for j in prange(1, Ny - 1):
        for i in range(1, Nx - 1):
            if solid[j, i]:
                u[j, i] = 0.0
                v[j, i] = 0.0
                continue
            u[j, i] -= dt * (p[j, i+1] - p[j, i-1]) / (2.0 * dx)
            v[j, i] -= dt * (p[j+1, i] - p[j-1, i]) / (2.0 * dy)


@njit(cache=True)
def _apply_bcs(u, v, u_inlet, Nx, Ny, solid):
    """Apply velocity boundary conditions."""
    # Bottom wall (j=0): no-slip
    for i in range(Nx):
        u[0, i] = 0.0
        v[0, i] = 0.0

    # Top wall (j=Ny-1): no-slip
    for i in range(Nx):
        u[Ny-1, i] = 0.0
        v[Ny-1, i] = 0.0

    # Inlet (i=0): parabolic u, v=0
    for j in range(Ny):
        u[j, 0] = u_inlet[j]
        v[j, 0] = 0.0

    # Outlet (i=Nx-1): zero-gradient
    for j in range(Ny):
        u[j, Nx-1] = u[j, Nx-2]
        v[j, Nx-1] = v[j, Nx-2]

    # Solid (cylinder): no-slip
    for j in range(Ny):
        for i in range(Nx):
            if solid[j, i]:
                u[j, i] = 0.0
                v[j, i] = 0.0


class FDSolver:
    """
    Finite Difference solver using Chorin's projection method.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Nx : int
        Grid cells in x-direction.
    Ny : int
        Grid cells in y-direction (if None, computed from aspect ratio).
    dt : float
        Time step (if None, computed from CFL condition).
    p_iters : int
        Number of pressure Poisson iterations per time step.
    """

    def __init__(self, Re, Nx=400, Ny=None, dt=None, p_iters=50):
        self.Re = Re

        self.Nx = Nx
        self.Ny = Ny if Ny else int(round(Nx * L_Y / L_X))

        self.dx = L_X / self.Nx
        self.dy = L_Y / self.Ny

        # Characteristic velocity: U_max of parabolic profile
        # Re = U_mean * D / nu, U_mean = (2/3)*U_max
        # Choose U_max = 1.0 (physical units), compute nu from Re
        self.U_max = 1.0
        self.U_mean = mean_velocity(self.U_max)
        self.nu = self.U_mean * CYL_D / Re

        # Time step from CFL + viscous stability
        dt_adv = 0.3 * min(self.dx, self.dy) / self.U_max
        dt_diff = 0.2 * min(self.dx, self.dy)**2 / self.nu
        self.dt = dt if dt else min(dt_adv, dt_diff)

        self.p_iters = p_iters

        # Build cylinder mask
        self.solid = _build_cylinder_mask(
            self.Nx, self.Ny, self.dx, self.dy)

        # Inlet velocity profile
        y_centers = (np.arange(self.Ny) + 0.5) * self.dy
        self.u_inlet = u_inlet_parabolic(y_centers, self.U_max)

        # Initialize fields
        self.u = np.zeros((self.Ny, self.Nx))
        self.v = np.zeros((self.Ny, self.Nx))
        self.p = np.zeros((self.Ny, self.Nx))

        # Set initial velocity to inlet profile
        for j in range(self.Ny):
            self.u[j, :] = self.u_inlet[j]
        self.u[self.solid] = 0.0

        self.history = []

        print(f"FD init: Nx={self.Nx}, Ny={self.Ny}, Re={Re}")
        print(f"  dx={self.dx:.5f}, dy={self.dy:.5f}, dt={self.dt:.6f}")
        print(f"  nu={self.nu:.6f}, U_max={self.U_max}, U_mean={self.U_mean:.4f}")
        print(f"  CFL_adv={self.U_max*self.dt/self.dx:.3f}, "
              f"CFL_diff={self.nu*self.dt/self.dx**2:.3f}")
        print(f"  Solid cells: {self.solid.sum()}")

    def step(self):
        """Perform one time step."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy, dt = self.dx, self.dy, self.dt

        # 1. Advection-diffusion -> intermediate velocity (u*, v*)
        u_star = self.u.copy()
        v_star = self.v.copy()
        _advect_diffuse(self.u, self.v, u_star, v_star,
                        Nx, Ny, dx, dy, dt, self.nu, self.solid)

        # Apply BCs to intermediate velocity
        _apply_bcs(u_star, v_star, self.u_inlet, Nx, Ny, self.solid)

        # 2. Pressure Poisson equation
        # RHS = (1/dt) * div(u*)
        rhs = np.zeros((Ny, Nx))
        rhs[1:-1, 1:-1] = (1.0 / dt) * (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2.0 * dx)
            + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2.0 * dy)
        )

        self.p = _pressure_poisson(
            self.p, rhs, Nx, Ny, dx, dy, self.solid,
            self.p_iters, 1.5)  # omega=1.5 for SOR

        # 3. Velocity correction: u^{n+1} = u* - dt * grad(p)
        self.u = u_star
        self.v = v_star
        _correct_velocity(self.u, self.v, self.p,
                          Nx, Ny, dx, dy, dt, self.solid)

        # 4. Apply BCs
        _apply_bcs(self.u, self.v, self.u_inlet, Nx, Ny, self.solid)

    def run(self, n_steps, report_interval=500):
        """Run simulation for n_steps."""
        # Probe behind cylinder
        pi = int((CYL_X + 5*CYL_R) / self.dx)
        pj = int((CYL_Y + CYL_R) / self.dy)
        pi = min(pi, self.Nx - 1)
        pj = min(pj, self.Ny - 1)

        t0 = time_mod.time()

        # Warm up numba JIT
        print("  Compiling (first step)...")
        self.step()
        self.history.append(self.v[pj, pi])
        compile_time = time_mod.time() - t0
        print(f"  JIT compiled in {compile_time:.1f}s")

        t0 = time_mod.time()
        for t in range(1, n_steps):
            self.step()
            self.history.append(self.v[pj, pi])

            if (t + 1) % report_interval == 0:
                speed = np.sqrt(self.u**2 + self.v**2)
                max_u = np.nanmax(speed)
                elapsed = time_mod.time() - t0
                print(f"  t={t+1:>7d}/{n_steps}  "
                      f"max|u|={max_u:.4f}  "
                      f"({(t+1)/elapsed:.0f} steps/s)")

                if np.any(np.isnan(self.u)) or max_u > 10.0:
                    print("  *** DIVERGED ***")
                    return self

        total = time_mod.time() - t0
        print(f"  Done in {total:.1f}s ({n_steps/total:.0f} steps/s)")
        return self

    def get_vorticity(self):
        """Compute vorticity field."""
        dv_dx = np.zeros_like(self.v)
        dv_dx[:, 1:-1] = (self.v[:, 2:] - self.v[:, :-2]) / (2*self.dx)

        du_dy = np.zeros_like(self.u)
        du_dy[1:-1, :] = (self.u[2:, :] - self.u[:-2, :]) / (2*self.dy)

        vort = dv_dx - du_dy
        vort[self.solid] = 0.0
        return vort

    def get_coords(self):
        """Return cell-center coordinate meshgrid."""
        x = (np.arange(self.Nx) + 0.5) * self.dx
        y = (np.arange(self.Ny) + 0.5) * self.dy
        return np.meshgrid(x, y)

    def get_strouhal(self):
        """Estimate Strouhal number from probe signal."""
        sig = np.array(self.history)
        n = len(sig)
        if n < 500:
            return None
        sig = sig[n//2:]
        sig -= sig.mean()
        if np.std(sig) < 1e-12:
            return None
        spec = np.abs(np.fft.rfft(sig))
        freq = np.fft.rfftfreq(len(sig), d=self.dt)
        spec[0] = 0
        f_dom = freq[np.argmax(spec)]
        St = f_dom * CYL_D / self.U_mean
        return St


# ============================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Re = 100
    Nx = 220
    n_steps = 3000

    print(f"=== FD Karman Vortex Street: Re={Re} ===\n")
    solver = FDSolver(Re=Re, Nx=Nx, p_iters=50)
    solver.run(n_steps, report_interval=1000)

    # Vorticity
    X, Y = solver.get_coords()
    vort = solver.get_vorticity()

    fig, ax = plt.subplots(figsize=(15, 3.5))
    vmax = max(np.percentile(np.abs(vort[~solver.solid]), 95), 1e-3)
    ax.pcolormesh(X, Y, vort, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  shading='auto')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(CYL_X + CYL_R*np.cos(theta),
            CYL_Y + CYL_R*np.sin(theta), color='gray', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f'FD Vorticity — Re={Re}, Nx={Nx}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('fd_vorticity.png', dpi=150)
    print("Saved fd_vorticity.png")

    # Velocity
    speed = np.sqrt(solver.u**2 + solver.v**2)
    fig2, ax2 = plt.subplots(figsize=(15, 3.5))
    ax2.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')
    ax2.fill(CYL_X + CYL_R*np.cos(theta),
             CYL_Y + CYL_R*np.sin(theta), color='gray', zorder=5)
    ax2.set_aspect('equal')
    ax2.set_title(f'FD |u| — Re={Re}')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('fd_velocity.png', dpi=150)
    print("Saved fd_velocity.png")

    # Probe
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(np.array(range(len(solver.history))) * solver.dt, solver.history)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('v at probe')
    ax3.set_title('FD Vortex shedding probe')
    plt.tight_layout()
    plt.savefig('fd_probe.png', dpi=150)
    print("Saved fd_probe.png")

    St = solver.get_strouhal()
    if St:
        print(f"Strouhal number: {St:.3f}")
    else:
        print("No periodic shedding detected yet")
