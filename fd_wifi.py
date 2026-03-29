import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
from numba import njit
from tqdm import tqdm

# --- 1. NUMBA HELPERS ---

@njit(cache=True)
def fast_stencil_assembly(nx, ny, cx, cy, diag):
    """Numba-accelerated matrix assembly."""
    N = nx * ny
    data = np.zeros(5 * N, dtype=np.complex128)
    rows = np.zeros(5 * N, dtype=np.int32)
    cols = np.zeros(5 * N, dtype=np.int32)
    counter = 0
    for j in range(ny):
        for i in range(nx):
            r_idx = j * nx + i
            data[counter], rows[counter], cols[counter] = diag[r_idx], r_idx, r_idx
            counter += 1
            if i > 0:
                data[counter], rows[counter], cols[counter] = cx, r_idx, r_idx - 1
                counter += 1
            if i < nx - 1:
                data[counter], rows[counter], cols[counter] = cx, r_idx, r_idx + 1
                counter += 1
            if j > 0:
                data[counter], rows[counter], cols[counter] = cy, r_idx, r_idx - nx
                counter += 1
            if j < ny - 1:
                data[counter], rows[counter], cols[counter] = cy, r_idx, r_idx + nx
                counter += 1
    return data[:counter], rows[:counter], cols[:counter]

@njit(cache=True)
def numba_calc_score(u_abs_sq, X, Y, targets, radius_sq):
    """Logarithmic scoring for colorblind-friendly balanced optimization."""
    total_score = 0.0
    ny, nx = u_abs_sq.shape
    for t_idx in range(len(targets)):
        tx, ty = targets[t_idx]
        sum_val, count = 0.0, 0
        for i in range(ny):
            for j in range(nx):
                dist_sq = (X[i, j] - tx) ** 2 + (Y[i, j] - ty) ** 2
                if dist_sq <= radius_sq:
                    sum_val += u_abs_sq[i, j]
                    count += 1
        if count > 0:
            total_score += 10.0 * np.log10((sum_val / count) + 1e-12)
    return total_score

# --- 2. WORKER ---

def evaluate_position(rx, ry, solve_lu, X, Y, targets):
    """Solve step using pre-factored LU."""
    # Width sigma = 0.2 is better for the 0.8GHz wavelength (~37.5cm)
    sigma = 0.2
    b = 1e5 * np.exp(-((X - rx) ** 2 + (Y - ry) ** 2).flatten() / (2 * sigma ** 2))
    u = solve_lu.solve(b).reshape(X.shape)
    score = numba_calc_score(np.abs(u) ** 2, X, Y, np.array(targets), 0.05 ** 2)
    return score, u, (rx, ry)

# --- 3. MAIN ---

def solve_wifi_optimized():
    Lx, Ly, res = 10.0, 8.0, 100
    nx, ny = int(Lx * res) + 1, int(Ly * res) + 1
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    f_hz = 0.8e9
    k0, wall_n, t_half = 2 * np.pi * f_hz / 3e8, 3*(2.5 + 0.5j), 0.075
    cx, cy = 1.0 / dx ** 2, 1.0 / dy ** 2

    # --- WALL MASKS ---
    # sim_mask is for physics (Interior only to respect Impedance BCs)
    sim_mask = np.zeros((ny, nx), dtype=bool)
    sim_mask[(Y >= 3 - t_half) & (Y <= 3 + t_half) & ((X <= 3) | ((X >= 4) & (X <= 6)) | (X >= 7))] = True
    sim_mask[(X >= 2.5 - t_half) & (X <= 2.5 + t_half) & (Y <= 2)] = True
    sim_mask[(X >= 7.0 - t_half) & (X <= 7.0 + t_half) & ((Y <= 1.5) | ((Y >= 2.5) & (Y <= 3)))] = True
    sim_mask[(X >= 6.0 - t_half) & (X <= 6.0 + t_half) & (Y >= 3)] = True

    # visual_mask is for plotting (Includes perimeter)
    visual_mask = sim_mask.copy()
    visual_mask[(Y <= 2*t_half)] = True           # Bottom
    visual_mask[(Y >= Ly - 2*t_half)] = True      # Top
    visual_mask[(X <= 2*t_half)] = True           # Left
    visual_mask[(X >= Lx - 2*t_half)] = True      # Right

    # One-time Factorization
    print("Building Matrix with Absorbing Boundaries...")
    diag = np.full((ny, nx), k0 ** 2, dtype=np.complex128)
    diag[sim_mask] = (k0 * wall_n) ** 2 # Physics only sees interior walls
    diag = -2 * cx - 2 * cy + diag.flatten()

    # Apply Impedance (Absorbing) BC to grid edges
    idx = np.arange(nx * ny).reshape((ny, nx))
    diag[idx[0, :]] += 1j * k0 / dy
    diag[idx[-1, :]] += 1j * k0 / dy
    diag[idx[:, 0]] += 1j * k0 / dx
    diag[idx[:, -1]] += 1j * k0 / dx

    data, r, c = fast_stencil_assembly(nx, ny, cx, cy, diag)
    A = sp.coo_matrix((data, (r, c)), shape=(nx * ny, nx * ny)).tocsc()
    solve_lu = spla.splu(A)

    # Candidates & Targets
    targets = [(1.0, 5.0), (2.0, 1.0), (9.0, 1.0), (9.0, 7.0)]
    x_c = np.linspace(0.5, 9.5, 16)
    y_c = np.linspace(0.5, 7.5, 13)
    # Filter candidates so router isn't in a wall
    candidates = [(rx, ry) for rx in x_c for ry in y_c if not visual_mask[int(ry * res), int(rx * res)]]

    # Parallel solving with tqdm
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(evaluate_position)(rx, ry, solve_lu, X, Y, targets)
        for rx, ry in tqdm(candidates, desc="Optimizing WiFi", unit="pos")
    )

    best_score, best_field, best_pos = max(results, key=lambda x: x[0])

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 9))
    db_map = 10 * np.log10(np.abs(best_field) ** 2 + 1e-10)

    # 1. Heatmap (Viridis for colorblind safety)
    im = plt.imshow(db_map, extent=[0, 10, 0, 8], origin='lower', cmap='viridis', vmin=-10, vmax=60)
    plt.colorbar(im, label="Signal Strength (dB)")

    # 2. Draw Walls in Solid Black (Using visual_mask)
    plt.contourf(X, Y, visual_mask, levels=[0.5, 1], colors='black')

    # 3. Target averaging regions
    for i, (tx, ty) in enumerate(targets):
        plt.scatter(tx, ty, marker='x', color='white', s=100, linewidths=2, zorder=5)
        circle = plt.Circle((tx, ty), 0.05, color='white', fill=False, linestyle='--', alpha=0.8, zorder=5)
        plt.gca().add_patch(circle)
        plt.text(tx + 0.15, ty + 0.15, f"T{i + 1}", color='white', fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    # 4. Best Router Position
    plt.scatter(*best_pos, marker='*', color='lime', s=500, edgecolors='black',
                label=f"Optimal Router Position: (X:{best_pos[0]:.2f}, Y:{best_pos[1]:.2f})", zorder=10)

    plt.title(f"WiFi Optimization (Absorbing Edges + Visual Perimeter)\nScore: {best_score:.2f} dB", fontsize=14)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.1)
    plt.show()

if __name__ == "__main__":
    solve_wifi_optimized()
