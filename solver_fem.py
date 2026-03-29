"""
solver_fem.py — Finite Element solver (NGSolve) for the Karman vortex street.

Semi-implicit time stepping with Taylor-Hood (P2/P1):
  - Implicit: viscous diffusion + pressure (Stokes operator, assembled once)
  - Explicit: convection (evaluated with previous velocity each step)

Pressure uniqueness: mean-zero constraint via Lagrange multiplier (NumberSpace).

"""

import sys
if "/usr/local/lib/python3.12/site-packages" not in sys.path:
    sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

import numpy as np
import time as time_mod

import ngsolve as ngs
from ngsolve import *
from netgen.geom2d import SplineGeometry

from geometry import (
    L_X, L_Y, CYL_X, CYL_Y, CYL_R, CYL_D,
    mean_velocity,
)


def build_mesh(max_h=0.02, refine_levels=2):
    """Build channel mesh with cylinder hole, refined near cylinder."""
    geo = SplineGeometry()

    # Channel boundary (counter-clockwise)
    p1 = geo.AddPoint(0, 0)
    p2 = geo.AddPoint(L_X, 0)
    p3 = geo.AddPoint(L_X, L_Y)
    p4 = geo.AddPoint(0, L_Y)

    geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="wall")
    geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="outlet")
    geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="wall")
    geo.Append(["line", p4, p1], leftdomain=1, rightdomain=0, bc="inlet")

    # Cylinder hole (clockwise)
    n_seg = 40
    cpts = []
    for k in range(n_seg):
        angle = 2 * np.pi * k / n_seg
        cpts.append(geo.AddPoint(
            CYL_X + CYL_R * np.cos(angle),
            CYL_Y + CYL_R * np.sin(angle)
        ))
    for k in range(n_seg):
        geo.Append(["line", cpts[k], cpts[(k + 1) % n_seg]],
                   leftdomain=0, rightdomain=1, bc="cyl")

    geo.SetMaterial(1, "fluid")
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=max_h))

    # Local refinement near cylinder
    for _ in range(refine_levels):
        for el in mesh.Elements(ngs.VOL):
            verts = [mesh[v].point for v in el.vertices]
            cx = sum(p[0] for p in verts) / len(verts)
            cy = sum(p[1] for p in verts) / len(verts)
            dist = np.sqrt((cx - CYL_X)**2 + (cy - CYL_Y)**2)
            mesh.SetRefinementFlag(el, dist < 4 * CYL_R)
        mesh.Refine()

    return mesh


class FEMSolver:
    """
    NGSolve FEM solver for 2D channel flow past a cylinder.

    Parameters
    ----------
    Re : float
        Reynolds number.
    max_h : float
        Max mesh element size.
    dt : float
        Time step.
    """

    def __init__(self, Re, max_h=0.025, dt=0.001):
        self.Re = Re
        self.dt = dt

        self.U_max = 1.0
        self.U_mean = mean_velocity(self.U_max)
        self.nu = self.U_mean * CYL_D / Re

        print(f"FEM init: Re={Re}, max_h={max_h}, dt={dt}")
        print(f"  nu={self.nu:.6f}, U_max={self.U_max}")

        # ---- Mesh ----
        print("  Building mesh...")
        self.mesh = build_mesh(max_h=max_h, refine_levels=1)
        print(f"  Mesh: {self.mesh.ne} elements, {self.mesh.nv} vertices")

        # ---- FE Spaces ----
        # Velocity: P2, Dirichlet on inlet, walls, cylinder
        # (outlet = do-nothing / natural BC)
        self.V = VectorH1(self.mesh, order=2,
                          dirichlet="inlet|wall|cyl")
        # Pressure: P1, no Dirichlet
        self.Q = H1(self.mesh, order=1)
        # Lagrange multiplier to pin mean pressure = 0
        self.N = NumberSpace(self.mesh)
        self.X = self.V * self.Q * self.N

        print(f"  DOFs: vel={self.V.ndof}, pres={self.Q.ndof}, "
              f"total={self.X.ndof}")

        # ---- Trial/test functions ----
        (u, p, lam), (v, q, mu) = self.X.TnT()

        # ---- Inlet profile ----
        self.u_inlet_cf = CoefficientFunction((
            4.0 * self.U_max * y * (L_Y - y) / L_Y**2,
            0.0
        ))

        # ---- Solution GridFunction ----
        self.gfu = GridFunction(self.X)
        self.velocity = self.gfu.components[0]
        self.pressure = self.gfu.components[1]

        # Set Dirichlet BCs
        self.velocity.Set(self.u_inlet_cf,
                          definedon=self.mesh.Boundaries("inlet"))
        self.velocity.Set(CoefficientFunction((0, 0)),
                          definedon=self.mesh.Boundaries("wall|cyl"))

        # ---- Previous velocity ----
        self.u_prev = GridFunction(self.V)
        # Initialize interior with parabolic profile
        self.u_prev.Set(self.u_inlet_cf)
        # Add perturbation behind cylinder to break symmetry
        pert = CoefficientFunction((
            0,
            0.001 * sin(4 * np.pi * y / L_Y)
        ))
        pert_gf = GridFunction(self.V)
        pert_gf.Set(pert)
        self.u_prev.vec.data += pert_gf.vec

        # ---- Assemble Stokes operator (implicit part, constant matrix) ----
        # (1/dt)(u,v) + nu(grad u, grad v) - (p, div v) - (q, div u)
        # + lam*(q,1) + mu*(p,1)   [mean pressure constraint]
        self.a = BilinearForm(self.X)
        self.a += (1.0 / dt) * InnerProduct(u, v) * dx
        self.a += self.nu * InnerProduct(Grad(u), Grad(v)) * dx
        self.a += -p * div(v) * dx
        self.a += -q * div(u) * dx
        self.a += lam * q * dx    # constraint: integral(p) = 0
        self.a += mu * p * dx
        self.a.Assemble()

        print("  Factorizing (pardiso)...")
        self.inv = self.a.mat.Inverse(self.X.FreeDofs(), inverse="umfpack")

        # Preallocate residual vector
        self.res = self.gfu.vec.CreateVector()

        self.history = []
        self.time = 0.0
        print("  FEM setup complete.")

    def step(self):
        """
        One semi-implicit time step.

        Implicit: (1/dt)*u + nu*laplacian(u) + grad(p) + pressure constraint
        Explicit: convection (u_prev . grad) u_prev
        """
        (u, p, lam), (v, q, mu) = self.X.TnT()

        # RHS = (1/dt)*u_prev - (u_prev . grad) u_prev
        rhs = LinearForm(self.X)
        rhs += (1.0 / self.dt) * InnerProduct(self.u_prev, v) * dx
        # Convection: -(grad(u_prev) * u_prev, v)
        rhs += -InnerProduct(Grad(self.u_prev) * self.u_prev, v) * dx
        rhs.Assemble()

        # Solve with Dirichlet values preserved:
        #   res = rhs - A * gfu_dirichlet
        #   gfu += inv * res  (only on free DOFs)
        self.res.data = rhs.vec - self.a.mat * self.gfu.vec
        self.gfu.vec.data += self.inv * self.res

        # Update convective velocity
        self.u_prev.vec.data = self.velocity.vec
        self.time += self.dt

    def run(self, n_steps, report_interval=200):
        """Run simulation for n_steps."""
        # Probe point behind cylinder
        px = CYL_X + 6 * CYL_R
        py = CYL_Y + CYL_R
        try:
            probe_mip = self.mesh(px, py)
        except:
            probe_mip = None
            print("  WARNING: probe point outside mesh")

        t0 = time_mod.time()
        for t in range(n_steps):
            self.step()

            # Record probe
            vy = 0.0
            if probe_mip is not None:
                try:
                    vy = self.velocity(probe_mip)[1]
                except:
                    pass
            self.history.append(vy)

            if (t + 1) % report_interval == 0:
                elapsed = time_mod.time() - t0
                # Quick max velocity estimate via Integrate
                speed_sq = InnerProduct(self.velocity, self.velocity)
                l2_sq = Integrate(speed_sq, self.mesh)
                area = Integrate(CoefficientFunction(1), self.mesh)
                rms_u = np.sqrt(l2_sq / area)

                print(f"  t={self.time:.3f}s  step={t+1}/{n_steps}  "
                      f"rms|u|={rms_u:.4f}  probe_vy={vy:.5f}  "
                      f"({(t+1)/elapsed:.1f} steps/s)")

                if np.isnan(rms_u) or rms_u > 10.0:
                    print("  *** DIVERGED ***")
                    return self

        total = time_mod.time() - t0
        print(f"  Done in {total:.1f}s ({n_steps/total:.1f} steps/s)")
        return self

    def get_numpy_fields(self, Nx=300, Ny=None):
        """Interpolate velocity to a regular grid for plotting."""
        if Ny is None:
            Ny = int(round(Nx * L_Y / L_X))

        xv = np.linspace(1e-6, L_X - 1e-6, Nx)
        yv = np.linspace(1e-6, L_Y - 1e-6, Ny)
        X, Y = np.meshgrid(xv, yv)

        ux = np.zeros((Ny, Nx))
        uy = np.zeros((Ny, Nx))

        for j in range(Ny):
            for i in range(Nx):
                if (xv[i] - CYL_X)**2 + (yv[j] - CYL_Y)**2 <= CYL_R**2:
                    continue
                try:
                    val = self.velocity(self.mesh(xv[i], yv[j]))
                    ux[j, i] = val[0]
                    uy[j, i] = val[1]
                except:
                    pass

        return X, Y, ux, uy

    def get_vorticity(self, Nx=300, Ny=None):
        """Compute vorticity on regular grid."""
        X, Y, ux, uy = self.get_numpy_fields(Nx, Ny)
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]

        duy_dx = np.zeros_like(uy)
        duy_dx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / (2 * dx)

        dux_dy = np.zeros_like(ux)
        dux_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / (2 * dy)

        return X, Y, duy_dx - dux_dy

    def get_strouhal(self):
        """Estimate Strouhal number from probe signal."""
        sig = np.array(self.history)
        n = len(sig)
        if n < 200:
            return None
        sig = sig[n // 2:]
        sig -= sig.mean()
        if np.std(sig) < 1e-12:
            return None
        spec = np.abs(np.fft.rfft(sig))
        freq = np.fft.rfftfreq(len(sig), d=self.dt)
        spec[0] = 0
        return freq[np.argmax(spec)] * CYL_D / self.U_mean


# ============================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Re = 100
    max_h = 0.03
    dt = 0.001
    n_steps = 8000  # 8 seconds - enough for vortex shedding

    print(f"=== FEM Karman Vortex Street: Re={Re} ===\n")
    solver = FEMSolver(Re=Re, max_h=max_h, dt=dt)
    solver.run(n_steps, report_interval=1000)

    # ---- Plots ----
    print("Interpolating to grid...")
    X, Y, vort = solver.get_vorticity(Nx=250)

    fig, ax = plt.subplots(figsize=(15, 3.5))
    nonzero = np.abs(vort[vort != 0])
    vmax = np.percentile(nonzero, 95) if len(nonzero) > 0 else 1.0
    vmax = max(vmax, 0.1)
    ax.pcolormesh(X, Y, vort, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  shading='auto')
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(CYL_X + CYL_R * np.cos(theta),
            CYL_Y + CYL_R * np.sin(theta), color='gray', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f'FEM Vorticity — Re={Re}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('fem_vorticity.png', dpi=150)
    print("Saved fem_vorticity.png")

    _, _, ux, uy = solver.get_numpy_fields(Nx=250)
    speed = np.sqrt(ux**2 + uy**2)
    fig2, ax2 = plt.subplots(figsize=(15, 3.5))
    ax2.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')
    ax2.fill(CYL_X + CYL_R * np.cos(theta),
             CYL_Y + CYL_R * np.sin(theta), color='gray', zorder=5)
    ax2.set_aspect('equal')
    ax2.set_title(f'FEM |u| — Re={Re}')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('fem_velocity.png', dpi=150)
    print("Saved fem_velocity.png")

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    time_arr = np.arange(len(solver.history)) * dt
    ax3.plot(time_arr, solver.history)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('v at probe')
    ax3.set_title('FEM Vortex shedding probe')
    plt.tight_layout()
    plt.savefig('fem_probe.png', dpi=150)
    print("Saved fem_probe.png")

    St = solver.get_strouhal()
    if St:
        print(f"Strouhal number: {St:.3f}")
    else:
        print("No periodic shedding detected")
