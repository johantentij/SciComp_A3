"""
solver_lbm.py — Lattice Boltzmann Method (D2Q9) for the Karman vortex street.

Numba-accelerated implementation with:
  - BGK collision
  - Bounce-back for cylinder and walls
  - Equilibrium inlet BC
  - Zero-gradient outlet
"""

import numpy as np
import time as time_mod
from numba import njit, prange
from geometry import (
    L_X, L_Y, CYL_D, CYL_R, CYL_X, CYL_Y,
    u_inlet_parabolic, mean_velocity,
)


# ============================================================
# D2Q9 constants (module-level for Numba access)
# ============================================================
_CX = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1], dtype=np.int32)
_CY = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1], dtype=np.int32)
_W  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


@njit(cache=True)
def _equilibrium_inplace(feq, rho, ux, uy, Ny, Nx):
    """Compute equilibrium distribution in-place."""
    for j in range(Ny):
        for i in range(Nx):
            usq = ux[j, i]**2 + uy[j, i]**2
            for q in range(9):
                cu = _CX[q] * ux[j, i] + _CY[q] * uy[j, i]
                feq[q, j, i] = _W[q] * rho[j, i] * (
                    1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq
                )


@njit(cache=True, parallel=True)
def _collide_stream_bc(f, f_new, rho, ux, uy, solid, u_inlet,
                        omega, Ny, Nx):
    """
    Combined collide + stream + boundary conditions in one pass.

    This is much faster than separate NumPy operations because:
    - No temporary array allocations
    - Single pass over memory
    - Numba parallel over rows
    """
    # 1. COLLISION + STREAM
    # For each fluid node, compute post-collision and place at destination.
    # For solid nodes, do bounce-back.

    # First zero out f_new
    for q in range(9):
        for j in prange(Ny):
            for i in range(Nx):
                f_new[q, j, i] = 0.0

    # Collide and stream
    for j in prange(Ny):
        for i in range(Nx):
            if solid[j, i]:
                # Bounce-back: send populations back to where they came from
                for q in range(9):
                    # The population that arrived at (j,i) from direction q
                    # gets reflected to opposite direction
                    oq = _OPP[q]
                    # It should go back to where it came from
                    ni = i - _CX[q]
                    nj = j - _CY[q]
                    if 0 <= ni < Nx and 0 <= nj < Ny:
                        f_new[oq, nj, ni] = f[q, j, i]
            else:
                # BGK collision
                usq = ux[j, i]**2 + uy[j, i]**2
                for q in range(9):
                    cu = _CX[q] * ux[j, i] + _CY[q] * uy[j, i]
                    feq = _W[q] * rho[j, i] * (
                        1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq
                    )
                    f_post = f[q, j, i] - omega * (f[q, j, i] - feq)

                    # Stream to destination
                    ni = i + _CX[q]
                    nj = j + _CY[q]
                    if 0 <= ni < Nx and 0 <= nj < Ny:
                        f_new[q, nj, ni] = f_post
                    # If out of bounds, the population is lost (absorbed)

    # 2. INLET BC (i=0): equilibrium with prescribed velocity
    for j in range(1, Ny - 1):
        r = rho[j, 0]
        if r < 0.01:
            r = 1.0
        usq = u_inlet[j]**2
        for q in range(9):
            cu = _CX[q] * u_inlet[j]
            f_new[q, j, 0] = _W[q] * r * (
                1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq
            )

    # 3. OUTLET BC (i=Nx-1): copy from i=Nx-2
    for q in range(9):
        for j in range(Ny):
            f_new[q, j, Nx-1] = f_new[q, j, Nx-2]

    # 4. WALL BCs are handled by solid mask (top/bottom rows are solid)


@njit(cache=True, parallel=True)
def _compute_macro(f, rho, ux, uy, solid, Ny, Nx):
    """Compute macroscopic density and velocity."""
    for j in prange(Ny):
        for i in range(Nx):
            if solid[j, i]:
                rho[j, i] = 1.0
                ux[j, i] = 0.0
                uy[j, i] = 0.0
                continue
            r = 0.0
            u = 0.0
            v = 0.0
            for q in range(9):
                r += f[q, j, i]
                u += _CX[q] * f[q, j, i]
                v += _CY[q] * f[q, j, i]
            if r > 1e-10:
                rho[j, i] = r
                ux[j, i] = u / r
                uy[j, i] = v / r
            else:
                rho[j, i] = 1.0
                ux[j, i] = 0.0
                uy[j, i] = 0.0


class LBMSolver:
    """
    Numba-accelerated LBM solver for 2D flow past a cylinder.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Ny : int
        Lattice nodes in y-direction.
    U_max_lbm : float
        Max inlet velocity in lattice units (keep <= 0.1).
    """

    def __init__(self, Re, Ny=120, U_max_lbm=0.05):
        self.Re = Re
        self.Ny = Ny
        self.Nx = int(round(Ny * L_X / L_Y))
        self.dx = L_Y / (Ny - 1)

        self.U_max = U_max_lbm
        self.U_mean = mean_velocity(U_max_lbm)
        self.D_lat = CYL_D / self.dx
        self.nu_lat = self.U_mean * self.D_lat / Re
        self.tau = 3.0 * self.nu_lat + 0.5
        self.omega = 1.0 / self.tau

        # Build solid mask (cylinder + top/bottom walls)
        x = np.arange(self.Nx) * self.dx
        y = np.arange(self.Ny) * self.dx
        X, Y = np.meshgrid(x, y)
        self.solid = ((X - CYL_X)**2 + (Y - CYL_Y)**2 <= CYL_R**2)
        self.solid[0, :] = True
        self.solid[-1, :] = True

        # Inlet profile
        y_phys = np.arange(self.Ny) * self.dx
        self.u_inlet = u_inlet_parabolic(y_phys, U_max_lbm)

        # Fields
        self.rho = np.ones((self.Ny, self.Nx))
        self.ux = np.zeros((self.Ny, self.Nx))
        self.uy = np.zeros((self.Ny, self.Nx))

        # Init velocity
        for j in range(self.Ny):
            self.ux[j, :] = self.u_inlet[j]
        self.ux[self.solid] = 0.0

        # Distribution functions
        self.f = np.zeros((9, self.Ny, self.Nx))
        _equilibrium_inplace(self.f, self.rho, self.ux, self.uy,
                             self.Ny, self.Nx)
        self.f_new = np.zeros_like(self.f)

        self.history = []

        print(f"LBM init: Nx={self.Nx}, Ny={self.Ny}, Re={Re}")
        print(f"  D_lat={self.D_lat:.1f}, tau={self.tau:.4f}, "
              f"nu_lat={self.nu_lat:.5f}")
        if self.tau < 0.52:
            print(f"  WARNING: tau close to 0.5, may be unstable")

    def step(self):
        """One LBM time step."""
        _collide_stream_bc(
            self.f, self.f_new, self.rho, self.ux, self.uy,
            self.solid, self.u_inlet, self.omega, self.Ny, self.Nx
        )
        # Swap buffers
        self.f, self.f_new = self.f_new, self.f

        # Compute macroscopic quantities
        _compute_macro(self.f, self.rho, self.ux, self.uy,
                       self.solid, self.Ny, self.Nx)

    def run(self, n_steps, report_interval=2000):
        """Run simulation."""
        # Probe location
        pi = int((CYL_X + 6*CYL_R) / self.dx)
        pj = int((CYL_Y + CYL_R) / self.dx)
        pi = min(pi, self.Nx - 1)
        pj = min(pj, self.Ny - 1)

        # JIT warmup
        print("  Compiling JIT...")
        t0 = time_mod.time()
        self.step()
        self.history.append(self.uy[pj, pi])
        print(f"  JIT compiled in {time_mod.time()-t0:.1f}s")

        t0 = time_mod.time()
        for t in range(1, n_steps):
            self.step()
            self.history.append(self.uy[pj, pi])

            if (t + 1) % report_interval == 0:
                max_u = np.sqrt(self.ux**2 + self.uy**2).max()
                elapsed = time_mod.time() - t0
                print(f"  t={t+1:>7d}/{n_steps}  "
                      f"max|u|={max_u:.5f}  "
                      f"rho=[{self.rho.min():.4f},{self.rho.max():.4f}]  "
                      f"({(t+1)/elapsed:.0f} steps/s)")
                if np.any(np.isnan(self.rho)) or max_u > 0.4:
                    print("  *** UNSTABLE ***")
                    return self

        total = time_mod.time() - t0
        print(f"  Done in {total:.1f}s ({n_steps/total:.0f} steps/s)")
        return self

    def get_vorticity(self):
        """Compute vorticity."""
        duy_dx = np.zeros_like(self.uy)
        duy_dx[:, 1:-1] = (self.uy[:, 2:] - self.uy[:, :-2]) / 2.0
        dux_dy = np.zeros_like(self.ux)
        dux_dy[1:-1, :] = (self.ux[2:, :] - self.ux[:-2, :]) / 2.0
        vort = duy_dx - dux_dy
        vort[self.solid] = 0.0
        return vort

    def get_coords(self):
        """Physical coordinate meshgrid."""
        x = np.arange(self.Nx) * self.dx
        y = np.arange(self.Ny) * self.dx
        return np.meshgrid(x, y)

    def get_strouhal(self):
        """Estimate Strouhal number."""
        sig = np.array(self.history)
        n = len(sig)
        if n < 1000:
            return None
        sig = sig[n//2:]
        sig -= sig.mean()
        if np.std(sig) < 1e-12:
            return None
        spec = np.abs(np.fft.rfft(sig))
        freq = np.fft.rfftfreq(len(sig))
        spec[0] = 0
        f_dom = freq[np.argmax(spec)]
        return f_dom * self.D_lat / self.U_mean


# ============================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Re = 100
    Ny = 120
    n_steps = 60000

    print(f"=== LBM Karman Vortex Street (Numba): Re={Re} ===\n")
    solver = LBMSolver(Re=Re, Ny=Ny, U_max_lbm=0.05)
    solver.run(n_steps, report_interval=10000)

    # Vorticity
    X, Y = solver.get_coords()
    vort = solver.get_vorticity()

    fig, ax = plt.subplots(figsize=(15, 3.5))
    fluid = ~solver.solid
    vmax = max(np.percentile(np.abs(vort[fluid]), 95), 1e-6)
    ax.pcolormesh(X, Y, vort, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  shading='auto')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(CYL_X + CYL_R*np.cos(theta),
            CYL_Y + CYL_R*np.sin(theta), color='gray', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f'LBM Vorticity — Re={Re}, Ny={Ny}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('lbm_vorticity.png', dpi=150)
    print("Saved lbm_vorticity.png")

    # Velocity
    speed = np.sqrt(solver.ux**2 + solver.uy**2)
    fig2, ax2 = plt.subplots(figsize=(15, 3.5))
    ax2.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')
    ax2.fill(CYL_X + CYL_R*np.cos(theta),
             CYL_Y + CYL_R*np.sin(theta), color='gray', zorder=5)
    ax2.set_aspect('equal')
    ax2.set_title(f'LBM |u| — Re={Re}')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('lbm_velocity.png', dpi=150)
    print("Saved lbm_velocity.png")

    # Probe
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(solver.history)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('v_y at probe')
    ax3.set_title('Vortex shedding probe')
    plt.tight_layout()
    plt.savefig('lbm_probe.png', dpi=150)
    print("Saved lbm_probe.png")

    St = solver.get_strouhal()
    if St:
        print(f"Strouhal number: {St:.3f}")
    else:
        print("No oscillation detected")
