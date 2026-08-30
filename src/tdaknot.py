"""
tdaknot.py - Topological Data Analysis based knot classification.

A second, independent method from knotclassify.py's classical (KMT +
Alexander polynomial) approach.

Builds a Vietoris-Rips complex on the full pairwise 3D distance matrix between all points on the curve (not just curve-adjacency), since knotting
is an extrinsic property. A simple closed curve's intrinsic topology is
always just a circle, knotted or not.

Validated (05_confinement_study.ipynb): reliable for
unconfined configurations (8/8 unknots -> 1 feature, 8/8 trefoils -> >1
feature, and full agreement with the classical classifier across a
simulated relaxation trajectory). Not reliable under confinement -
compression brings distant-along-strand points close in 3D space without
threading, causing false positives. See notebook 5 for more.
"""

import numpy as np
import gudhi as gd


def select_threshold(lifetimes, min_features=1, eps=1e-9):
    """Automatic persistence threshold via the largest-gap heuristic in log space."""
    sl = np.sort(lifetimes)[::-1]
    if len(sl) <= min_features:
        return sl[-1] * 0.99 if len(sl) else 0.0
    log_sl = np.log(sl + eps)
    gaps = log_sl[:-1] - log_sl[1:]
    cut_idx = np.argmax(gaps)
    return np.exp((log_sl[cut_idx] + log_sl[cut_idx + 1]) / 2) - eps


def compute_h1_persistence(positions_3d, max_edge_length=None, max_dimension=2):
    if max_edge_length is None:
        max_edge_length = np.ptp(positions_3d, axis=0).max()
    rips = gd.RipsComplex(points=positions_3d, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dimension)
    persistence = st.persistence()
    lifetimes = np.array([d - b for dim, (b, d) in persistence
                           if dim == 1 and d != float('inf')])
    return lifetimes


def entanglement_score(positions_3d, max_edge_length=None):
    lifetimes = compute_h1_persistence(positions_3d, max_edge_length)
    if len(lifetimes) <= 1:
        return len(lifetimes), lifetimes
    threshold = select_threshold(lifetimes)
    n_significant = int((lifetimes > threshold).sum())
    return n_significant, lifetimes


def is_knotted(positions_3d, max_edge_length=None):
    n, _ = entanglement_score(positions_3d, max_edge_length)
    return n > 1
