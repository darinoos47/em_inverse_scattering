# utils/green_generator.py

import numpy as np
from scipy.special import hankel2, jv

def generate_green_matrices(D=2, R=3, epsilon_b=1.0, N=40, Nr=36, f=4e8):
    """
    Generate Gs, Gd, and Ei for 2D inverse scattering.
    Returns: Gs [Nr, M], Gd [M, M], Ei [M, Nr]
    """

    c = 3e8  # Speed of light
    k0 = np.sqrt(epsilon_b) * 2 * np.pi * f / c
    M = N * N # Total number of internal cells
    Ra = (D / N) / np.sqrt(np.pi) # Equivalent radius of a square cell

    # Grid coordinates (centers of the square cells)
    # These coordinates are used for both internal (object) points and source points
    xq, yq = np.meshgrid(
        np.linspace(-D/2 + D/(2*N), D/2 - D/(2*N), N),
        np.linspace(-D/2 + D/(2*N), D/2 - D/(2*N), N)
    )
    yq = np.flipud(yq)  # Match MATLAB mesh ordering (optional, but good for consistency)
    Xi0 = xq.flatten(order='F') # Flattened x-coordinates of internal cells
    Yi0 = yq.flatten(order='F') # Flattened y-coordinates of internal cells

    # Receiver positions (measurement points)
    theta = np.linspace(0, 360, Nr, endpoint=False) # Angles for receiver positions
    Xr0 = R * np.cos(np.deg2rad(theta)) # x-coordinates of receivers
    Yr0 = R * np.sin(np.deg2rad(theta)) # y-coordinates of receivers

    # --- Gs computation (Green's function from internal cells to receivers) ---
    # Gs is a matrix where Gs[i,j] represents the field at receiver i due to a source at internal cell j
    Xi_gs = np.tile(Xi0, (Nr, 1))   # [Nr, M] - Repeat internal x-coords for each receiver
    Yi_gs = np.tile(Yi0, (Nr, 1))   # [Nr, M] - Repeat internal y-coords for each receiver
    Xr_gs = np.repeat(Xr0[:, np.newaxis], M, axis=1) # [Nr, M] - Repeat receiver x-coords for each internal cell
    Yr_gs = np.repeat(Yr0[:, np.newaxis], M, axis=1) # [Nr, M] - Repeat receiver y-coords for each internal cell

    # Calculate distances between internal cells and receivers
    r_gs = k0 * np.sqrt((Xi_gs - Xr_gs)**2 + (Yi_gs - Yr_gs)**2)
    
    # Compute Gs using Hankel function of the second kind, order 0
    # The factor (k0**2) * (D/N)**2 / (4j) comes from the 2D Green's function formulation
    Gs = (k0**2) * (D/N)**2 / (4j) * hankel2(0, r_gs)

    # --- Gd computation (Green's function from internal cells to other internal cells) ---
    # Gd is a matrix where Gd[i,j] represents the field at internal cell i due to a source at internal cell j
    Xm_gd = np.tile(Xi0, (M, 1))  # [M, M] - Repeat internal x-coords for each target internal cell
    Ym_gd = np.tile(Yi0, (M, 1))  # [M, M] - Repeat internal y-coords for each target internal cell
    Xn_gd = Xm_gd.T               # [M, M] - Transposed for source internal cells
    Yn_gd = Ym_gd.T               # [M, M] - Transposed for source internal cells

    # Calculate distances between internal cells
    r_gd = k0 * np.sqrt((Xm_gd - Xn_gd)**2 + (Ym_gd - Yn_gd)**2)
    
    # Initialize Gd matrix
    Gd = np.zeros((M, M), dtype=np.complex64)

    # Compute off-diagonal elements of Gd
    # Create a mask for off-diagonal elements (where r_gd is not zero)
    off_diagonal_mask = r_gd != 0
    Gd[off_diagonal_mask] = (k0**2) * (D/N)**2 / (4j) * hankel2(0, r_gd[off_diagonal_mask])

    # Diagonal correction (self-term) for Gd
    # This is the analytical solution for the integral of the Green's function over a square cell
    # when the observation point is within the source cell.
    # This formula is specific to the Method of Moments (MoM) for square cells.
    for i in range(M):
        Gd[i, i] = (2*np.pi / (4j)) * (k0 * Ra * hankel2(1, k0 * Ra) - 1j * 2/np.pi)

    # --- Incident field Ei (field at internal cells due to incident waves) ---
    # Ei is a matrix where Ei[i,j] is the incident field at internal cell i due to incident wave j
    Ei = np.zeros((M, Nr), dtype=np.complex64) # Nr is used here as the number of incident waves
    # Calculate the angular step size based on the number of incident waves
    angular_step = 360.0 / Nr
    for i in range(Nr): # Iterate through each incident wave direction
        angle_rad = np.deg2rad(i * angular_step) # Calculate angle for uniform distribution
        # Plane wave incident field: exp(j * k0 * (x*cos(angle) + y*sin(angle)))
        Ei[:, i] = np.exp(1j * k0 * (Xi0 * np.cos(angle_rad) + Yi0 * np.sin(angle_rad)))

    return Gs.astype(np.complex64), Gd.astype(np.complex64), Ei.astype(np.complex64)


