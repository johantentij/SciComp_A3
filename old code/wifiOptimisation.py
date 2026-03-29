import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- 1. Grid and Domain Setup ---
width, height = 10.0, 8.0
h = 0.025  # Grid spacing in meters (5 cm resolution)
Nx = int(width / h) + 1
Ny = int(height / h) + 1

# Create coordinate matrices
X, Y = np.meshgrid(np.linspace(0, width, Nx), 
                   np.linspace(0, height, Ny), 
                   indexing='ij')

# --- 2. Physics Parameters ---
f_scaled = 8e8  # Scaled frequency: 0.8 GHz
c = 3e8           # Speed of light
k_base = 2 * np.pi * f_scaled / c  # Base wavenumber in air
K = np.ones((Nx, Ny), dtype=complex) * k_base

# Wall definitions (from your encoded layout)
dWall = 0.15
rWall = 0.5 * dWall
outerWalls = [
    [0, 0, 10, dWall],
    [0, dWall, dWall, 8 - dWall],
    [10 - dWall, dWall, 10, 8 - dWall],
    [0, 8 - dWall, 10, 8]
]
walls = [
    [2.5 - rWall, dWall, 2.5 + rWall, 2],
    [7 - rWall, dWall, 7 + rWall, 1.5],
    [7 - rWall, 2.5, 7 + rWall, 3],
    [dWall, 3 - rWall, 3, 3 + rWall],
    [4, 3 - rWall, 6, 3 + rWall],
    [7, 3 - rWall, 10 - dWall, 3 + rWall],
    [6 - rWall, 3, 6 + rWall, 8 - dWall],
]
all_walls = walls + outerWalls

# Apply material refractive index to the wavenumber K [cite: 376]
for w in all_walls:
    x0, y0, x1, y1 = w
    i0, i1 = max(0, int(x0 / h)), min(Nx, int(x1 / h) + 1)
    j0, j1 = max(0, int(y0 / h)), min(Ny, int(y1 / h) + 1)
    K[i0:i1, j0:j1] = k_base * (2.5 + 0.5j)

# --- 3. Source Term (Router) ---
xr, yr = 2.5, 5.5  # Router position
A_src = 1e4        # Amplitude [cite: 362]
sigma = 0.2        # Pulse width [cite: 362]
# Gaussian pulse [cite: 361]
F = A_src * np.exp(-((X - xr)**2 + (Y - yr)**2) / (2 * sigma**2))

# --- 4. Matrix Assembly ---
N = Nx * Ny
def idx(i, j): 
    return i * Ny + j

print("Assembling sparse matrix...")
A = sp.lil_matrix((N, N), dtype=complex)
b = np.zeros(N, dtype=complex)

for i in range(Nx):
    for j in range(Ny):
        row = idx(i, j)
        k_ij = K[i, j]
        
        # 1st-order absorbing boundaries for the outer edges
        if i == 0:
            A[row, row] = 1j * k_ij * h - 1
            A[row, idx(1, j)] = 1
        elif i == Nx - 1:
            A[row, row] = 1j * k_ij * h - 1
            A[row, idx(Nx-2, j)] = 1
        elif j == 0:
            A[row, row] = 1j * k_ij * h - 1
            A[row, idx(i, 1)] = 1
        elif j == Ny - 1:
            A[row, row] = 1j * k_ij * h - 1
            A[row, idx(i, Ny-2)] = 1
        else:
            # Internal 5-point stencil
            A[row, row] = (h**2) * (k_ij**2) - 4
            A[row, idx(i+1, j)] = 1
            A[row, idx(i-1, j)] = 1
            A[row, idx(i, j+1)] = 1
            A[row, idx(i, j-1)] = 1
            b[row] = F[i, j] * (h**2)

# --- 5. Solve the System ---
print("Solving linear system...")
# Convert to Compressed Sparse Row (CSR) format for fast solving
A_csr = A.tocsr()
u_flat = spla.spsolve(A_csr, b)
U = u_flat.reshape((Nx, Ny))

# --- 6. Process and Plot Results ---
# Convert complex field to decibels (dB)
magnitude = np.abs(U)
magnitude[magnitude < 1e-10] = 1e-10  # Prevent log(0) warnings
signal_db = 20 * np.log10(magnitude)

plt.figure(figsize=(12, 8))
# Plot the signal heat map
plt.pcolormesh(X, Y, signal_db, shading='auto', cmap='jet', vmin=-40, vmax=0)
cbar = plt.colorbar()
cbar.set_label('Signal Strength (dB)')

# Overlay the walls
for w in all_walls:
    x0, y0, x1, y1 = w
    plt.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color='black', alpha=0.8)

# Mark the router
plt.plot(xr, yr, 'w*', markersize=15, markeredgecolor='k', label='WiFi Router')

plt.title('Approximated WiFi Signal Coverage (0.8 GHz)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.legend(loc='upper right')
plt.axis('equal')
plt.tight_layout()
plt.show()