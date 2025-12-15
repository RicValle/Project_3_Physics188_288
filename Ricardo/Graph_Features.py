
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

LATTICE_TYPES = ("SC", "BCC", "FCC", "diamond")

@dataclass(frozen=True)
class GraphSpec:
    lattice_type: str
    size: int
    a: float
    prob: float
    nearest_neighbor: bool
    A: float
    alpha: float
    t_max: float | None = None 

def one_hot(value: str, choices: Tuple[str, ...]) -> np.ndarray:
    x = np.zeros(len(choices), dtype=float)
    x[choices.index(value)] = 1.0
    return x

def minimal_image_displacements(coords: np.ndarray, box_size: float) -> np.ndarray:
    """
    coords: (N,3)
    returns: disp (N,N,3) with minimal image convention
    """
    diff = coords[:, None, :] - coords[None, :, :]
    diff = diff - box_size * np.round(diff / box_size)
    return diff

def hop_length_moments(coords: np.ndarray, W: np.ndarray, box_size: float) -> Dict[str, float]:
    """
    Compute hop-length statistics induced by the rate matrix W.
    Treat outgoing rates from node j as weights over i != j.
    """
    N = coords.shape[0]
    disp = minimal_image_displacements(coords, box_size)
    dist = np.linalg.norm(disp, axis=2)  # (N,N)

    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)

    lam = np.sum(W_off, axis=0)  # (N,)
    eps = 1e-15
    lam_safe = np.maximum(lam, eps)

    mean_r_per_j = np.sum(W_off * dist, axis=0) / lam_safe
    mean_r2_per_j = np.sum(W_off * dist**2, axis=0) / lam_safe

    return {
        "mean_hop_len": float(np.mean(mean_r_per_j)),
        "std_hop_len": float(np.std(mean_r_per_j)),
        "mean_hop_len2": float(np.mean(mean_r2_per_j)),
        "mean_escape_rate": float(np.mean(lam)),
        "std_escape_rate": float(np.std(lam)),
    }

def degree_stats_from_W(W: np.ndarray, tol: float = 0.0) -> Dict[str, float]:
    """
    Define (out)degree as number of i != j with W[i,j] > tol.
    """
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    deg = np.sum(W_off > tol, axis=0).astype(float)
    return {
        "mean_degree": float(np.mean(deg)),
        "std_degree": float(np.std(deg)),
        "min_degree": float(np.min(deg)),
        "max_degree": float(np.max(deg)),
    }

def build_structured_features(
    spec: GraphSpec,
    coords: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      phi: 1D feature vector
      names: aligned feature names
    If coords and W are provided, includes realized-graph features.
    """
    
    phi_parts: List[np.ndarray] = []
    names: List[str] = []

    lt = one_hot(spec.lattice_type, LATTICE_TYPES)
    phi_parts.append(lt)
    names += [f"lattice_{k}" for k in LATTICE_TYPES]

    box_size = spec.a * (2 * spec.size + 1)
    core = np.array([
        spec.size,
        spec.a,
        box_size,
        spec.prob,
        float(spec.nearest_neighbor),
        spec.A,
        0.0 if spec.nearest_neighbor else spec.alpha,
    ], dtype=float)
    phi_parts.append(core)
    names += ["size", "a", "box_size", "prob", "is_nn", "A", "alpha_eff"]

    if coords is not None:
        N = float(coords.shape[0])
        phi_parts.append(np.array([N], dtype=float))
        names += ["N_nodes"]

    if coords is not None and W is not None:

        deg_stats = degree_stats_from_W(W, tol=0.0)
        hop_stats = hop_length_moments(coords, W, box_size)

        extra = np.array([
            deg_stats["mean_degree"],
            deg_stats["std_degree"],
            deg_stats["min_degree"],
            deg_stats["max_degree"],
            hop_stats["mean_hop_len"],
            hop_stats["std_hop_len"],
            hop_stats["mean_hop_len2"],
            hop_stats["mean_escape_rate"],
            hop_stats["std_escape_rate"],
        ], dtype=float)

        phi_parts.append(extra)
        names += [
            "mean_degree", "std_degree", "min_degree", "max_degree",
            "mean_hop_len", "std_hop_len", "mean_hop_len2",
            "mean_escape_rate", "std_escape_rate",
        ]

    phi = np.concatenate(phi_parts, axis=0)
    return phi, names
