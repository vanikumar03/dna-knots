"""
dnasim.py - Coarse-grained bead-spring DNA polymer simulation.

Implements a closed (ring) bead-spring chain under overdamped Langevin
(Brownian) dynamics, with:
    - FENE bonds (chain connectivity, finite extensibility)
    - WCA excluded volume (purely repulsive - prevents the chain passing
      through itself)
    - Kratky-Porod bending stiffness (controls persistence length /
      flexibility)
    - optional spherical confinement potential

All quantities are in standard reduced (Lennard-Jones) units: sigma = 1
(bead diameter / length unit), epsilon = 1 (energy unit), k_B*T = 1.

Every force function here has a matching energy function, verified against
a finite-difference gradient of its own energy in
01_polymer_simulator.ipynb before being trusted.
"""

import os
import numpy as np




# FENE bond potential (chain connectivity)


def fene_energy(positions, K=30.0, R0=1.5):
    N = len(positions)
    nxt = np.roll(positions, -1, axis=0)
    bonds = nxt - positions
    r = np.linalg.norm(bonds, axis=1)
    r = np.clip(r, 1e-10, R0 - 1e-6)
    return np.sum(-0.5 * K * R0**2 * np.log(1 - (r / R0) ** 2))


def fene_force(positions, K=30.0, R0=1.5):
    """Force F_i = [K/(1-(r/R0)^2)] * bond_i (using the raw bond vector -
    not divided by r again; the r's from dE/dr and dr/dx cancel exactly)."""
    N = len(positions)
    nxt = np.roll(positions, -1, axis=0)
    bonds = nxt - positions
    r = np.linalg.norm(bonds, axis=1)
    r_safe = np.clip(r, 1e-10, R0 - 1e-6)
    mag = K / (1 - (r_safe / R0) ** 2)
    bond_force = mag[:, None] * bonds

    forces = np.zeros_like(positions)
    forces += bond_force
    forces -= np.roll(bond_force, 1, axis=0)
    return forces





# WCA excluded volume

WCA_CUTOFF = 2 ** (1 / 6)


def wca_energy(positions, epsilon=1.0, sigma=1.0):
    N = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    cutoff = WCA_CUTOFF * sigma
    mask = dist < cutoff
    r = dist[mask]
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    e = 4 * epsilon * (sr12 - sr6) + epsilon
    return 0.5 * np.sum(e)


def wca_force(positions, epsilon=1.0, sigma=1.0):
    N = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    cutoff = WCA_CUTOFF * sigma
    mask = dist < cutoff
    forces = np.zeros_like(positions)
    if not mask.any():
        return forces
    r = dist[mask]
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    fmag = 24 * epsilon * (2 * sr12 - sr6) / r
    unit = diff[mask] / dist[mask][:, None]
    contrib = fmag[:, None] * unit
    idx_i, idx_j = np.where(mask)
    np.add.at(forces, idx_i, contrib)
    return forces





# Kratky-Porod bending potential


def bending_energy(positions, kappa=0.0):
    if kappa == 0.0:
        return 0.0
    N = len(positions)
    prev = np.roll(positions, 1, axis=0)
    nxt = np.roll(positions, -1, axis=0)
    b1 = positions - prev
    b2 = nxt - positions
    n1 = np.linalg.norm(b1, axis=1)
    n2 = np.linalg.norm(b2, axis=1)
    cos_theta = np.sum(b1 * b2, axis=1) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return kappa * np.sum(1 - cos_theta)


def bending_force(positions, kappa=0.0):
    """Gradient of (1-cos(theta_i)) w.r.t. each of the 3 beads involved."""
    if kappa == 0.0:
        return np.zeros_like(positions)
    N = len(positions)
    forces = np.zeros_like(positions)
    prev_idx = np.roll(np.arange(N), 1)
    next_idx = np.roll(np.arange(N), -1)
    p_prev = positions[prev_idx]
    p_curr = positions
    p_next = positions[next_idx]
    b1 = p_curr - p_prev
    b2 = p_next - p_curr
    n1 = np.linalg.norm(b1, axis=1, keepdims=True)
    n2 = np.linalg.norm(b2, axis=1, keepdims=True)
    u1 = b1 / n1
    u2 = b2 / n2
    cos_theta = np.sum(u1 * u2, axis=1, keepdims=True)
    d_prev = -(u2 - cos_theta * u1) / n1
    d_next = (u1 - cos_theta * u2) / n2
    d_curr = -(d_prev + d_next)
    np.add.at(forces, prev_idx, kappa * d_prev)
    forces += kappa * d_curr
    np.add.at(forces, next_idx, kappa * d_next)
    return forces






# Spherical confinement


def confinement_energy(positions, radius=None, strength=50.0):
    if radius is None:
        return 0.0
    r = np.linalg.norm(positions, axis=1)
    overshoot = np.clip(r - radius, 0, None)
    return np.sum(0.5 * strength * overshoot ** 2)


def confinement_force(positions, radius=None, strength=50.0):
    if radius is None:
        return np.zeros_like(positions)
    r = np.linalg.norm(positions, axis=1)
    overshoot = np.clip(r - radius, 0, None)
    r_safe = np.clip(r, 1e-10, None)
    fmag = -strength * overshoot / r_safe
    return fmag[:, None] * positions







# Combined energy / force


def total_energy(positions, K=30.0, R0=1.5, epsilon=1.0, sigma=1.0,
                  kappa=0.0, confinement_radius=None, confinement_strength=50.0):
    return (
        fene_energy(positions, K, R0)
        + wca_energy(positions, epsilon, sigma)
        + bending_energy(positions, kappa)
        + confinement_energy(positions, confinement_radius, confinement_strength)
    )


def total_force(positions, K=30.0, R0=1.5, epsilon=1.0, sigma=1.0,
                 kappa=0.0, confinement_radius=None, confinement_strength=50.0):
    return (
        fene_force(positions, K, R0)
        + wca_force(positions, epsilon, sigma)
        + bending_force(positions, kappa)
        + confinement_force(positions, confinement_radius, confinement_strength)
    )





# Langevin (Brownian / overdamped) dynamics integrator


def langevin_step(positions, dt, gamma=1.0, kT=1.0, rng=None, **force_kwargs):
    if rng is None:
        rng = np.random.default_rng()
    F = total_force(positions, **force_kwargs)
    noise = rng.normal(size=positions.shape)
    D = kT / gamma
    return positions + (dt / gamma) * F + np.sqrt(2 * D * dt) * noise


def run_simulation(positions0, n_steps, dt, gamma=1.0, kT=1.0, seed=0,
                    record_every=50, **force_kwargs):
    rng = np.random.default_rng(seed)
    positions = positions0.copy()
    trajectory = [positions.copy()]
    for step in range(1, n_steps + 1):
        positions = langevin_step(positions, dt, gamma, kT, rng, **force_kwargs)
        if step % record_every == 0:
            trajectory.append(positions.copy())
    return trajectory





# Initial configurations


def make_ring(N, bond_length=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    circumference = N * bond_length
    radius = circumference / (2 * np.pi)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(N)], axis=1)
    positions += rng.normal(scale=noise, size=positions.shape)
    return positions


def make_trefoil_seed(N, scale=1.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    positions = np.stack([x, y, z], axis=1) * scale
    bond_lengths = np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=1)
    positions *= 1.0 / bond_lengths.mean()
    positions += rng.normal(scale=0.02, size=positions.shape)
    return positions





# Diagnostics


def radius_of_gyration(positions):
    center = positions.mean(axis=0)
    return np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1)))


def bond_length_stats(positions):
    nxt = np.roll(positions, -1, axis=0)
    lengths = np.linalg.norm(nxt - positions, axis=1)
    return lengths.mean(), lengths.std(), lengths.min(), lengths.max()


def min_pairwise_distance(positions):
    N = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist.min()
