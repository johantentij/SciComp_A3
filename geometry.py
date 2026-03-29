"""
geometry.py — Shared geometry and parameter definitions for the Kármán vortex street.

Channel: 2.2m x 0.41m (H = 0.41m)
Cylinder: diameter D = 0.1m, center at (0.2, 0.2)
Boundary conditions:
  - Inlet (left):   parabolic velocity profile
  - Outlet (right):  zero-gradient (do-nothing)
  - Top/bottom walls: no-slip (u=v=0)
  - Cylinder surface: no-slip (u=v=0)
"""

import numpy as np


# ============================================================
# Physical domain parameters
# ============================================================
L_X = 2.2       # channel length [m]
L_Y = 0.41      # channel height [m]
CYL_X = 0.2     # cylinder center x [m]
CYL_Y = 0.2     # cylinder center y [m]
CYL_D = 0.1     # cylinder diameter [m]
CYL_R = CYL_D / 2.0  # cylinder radius [m]


def u_inlet_parabolic(y, U_max):
    """
    Parabolic inlet velocity profile (Poiseuille-like).

    u(y) = 4 * U_max * y * (L_Y - y) / L_Y^2

    This gives u=0 at y=0 and y=L_Y, and u=U_max at y=L_Y/2.
    The mean velocity is (2/3)*U_max.

    Parameters
    ----------
    y : array_like
        y-coordinates of the inlet points.
    U_max : float
        Maximum centerline velocity.

    Returns
    -------
    u : array_like
        Horizontal velocity at each y-coordinate.
    """
    return 4.0 * U_max * y * (L_Y - y) / (L_Y ** 2)


def mean_velocity(U_max):
    """Mean velocity of the parabolic profile: U_mean = (2/3) * U_max."""
    return (2.0 / 3.0) * U_max


def Re_to_U_max(Re, nu):
    """
    Compute U_max from Reynolds number.

    Re = U_mean * D / nu = (2/3) * U_max * D / nu
    => U_max = (3/2) * Re * nu / D
    """
    return 1.5 * Re * nu / CYL_D


def compute_nu(Re, U_max):
    """
    Compute kinematic viscosity from Re and U_max.

    Re = U_mean * D / nu  =>  nu = U_mean * D / Re = (2/3)*U_max*D / Re
    """
    U_mean = mean_velocity(U_max)
    return U_mean * CYL_D / Re


def is_inside_cylinder(x, y):
    """
    Check whether points (x, y) are inside the cylinder.

    Parameters
    ----------
    x, y : array_like
        Coordinates (can be scalars, 1D, or 2D arrays).

    Returns
    -------
    mask : bool array
        True where the point is inside (or on) the cylinder.
    """
    return (x - CYL_X) ** 2 + (y - CYL_Y) ** 2 <= CYL_R ** 2


def make_grid(Nx, Ny):
    """
    Create a uniform Cartesian grid for the channel.

    Parameters
    ----------
    Nx : int
        Number of cells in x-direction.
    Ny : int
        Number of cells in y-direction.

    Returns
    -------
    x : 1D array of shape (Nx+1,)
    y : 1D array of shape (Ny+1,)
    dx : float
    dy : float
    X : 2D array of shape (Ny+1, Nx+1)   (meshgrid, indexing='ij' on y,x)
    Y : 2D array of shape (Ny+1, Nx+1)
    """
    dx = L_X / Nx
    dy = L_Y / Ny
    x = np.linspace(0, L_X, Nx + 1)
    y = np.linspace(0, L_Y, Ny + 1)
    X, Y = np.meshgrid(x, y)  # shape (Ny+1, Nx+1)
    return x, y, dx, dy, X, Y


def cylinder_mask_grid(X, Y):
    """
    Boolean mask for grid nodes inside the cylinder.

    Parameters
    ----------
    X, Y : 2D arrays from meshgrid.

    Returns
    -------
    mask : bool array, same shape as X.
    """
    return is_inside_cylinder(X, Y)


# ============================================================
# Lattice Boltzmann specific helpers
# ============================================================
def cylinder_mask_lbm(Nx, Ny):
    """
    Boolean mask for LBM lattice nodes inside the cylinder.

    In LBM the lattice coordinates are integers.
    We map lattice (i, j) -> physical (x, y) = (i*dx, j*dy).

    Parameters
    ----------
    Nx, Ny : int
        Number of lattice nodes in x and y directions.

    Returns
    -------
    mask : bool array of shape (Ny, Nx)
        True where the node is inside the cylinder.
    """
    dx = L_X / (Nx - 1)
    dy = L_Y / (Ny - 1)
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y)  # shape (Ny, Nx)
    return is_inside_cylinder(X, Y)
