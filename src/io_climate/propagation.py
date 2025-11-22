# src/io_climate/propagation.py

from typing import Dict, Any, Tuple
import numpy as np


def propagate_once(
    Z: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    globsec_of: np.ndarray,
    FD_eff: np.ndarray,
    X_cap: np.ndarray,
    sp: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Single iteration of the propagation algorithm:

        (FD_eff, X_cap) -> constraints, inventories, γ-reallocation -> (Z_new, X_new)

    Parameters
    ----------
    Z : (n, n)
        Intermediate-use matrix (producer i -> user j).
    A : (n, n)
        Technical coefficients matrix.
    L : (n, n)
        Leontief inverse.
    globsec_of : (n,)
        Global sector id for each country–sector.
    FD_eff : (n,)
        Effective final demand vector for this iteration.
    X_cap : (n,)
        Capacity vector X * (1 - sp).
    sp : (n,)
        Capacity shocks in [0,1].
    gamma : float
        Trade reallocation capacity in [0,1].

    Returns
    -------
    Z_new : (n, n)
        New intermediate-use matrix after γ-reallocation.
    X_new : (n,)
        New gross output vector.
    aux : dict
        Diagnostics (r, s, inventories, etc.).
    """
    n = Z.shape[0]
    S = int(globsec_of.max()) + 1

    FD_eff = np.asarray(FD_eff, dtype=float).reshape(-1)
    X_cap = np.asarray(X_cap, dtype=float).reshape(-1)
    sp = np.asarray(sp, dtype=float).reshape(-1)

    # 1. Demand-only output (ignoring capacity constraints)
    X_dem = L @ FD_eff

    # 2. Rationing factors r_i = min(1, X_cap_i / X_dem_i)
    r = np.ones(n, dtype=float)
    mask_dem = X_dem > 0.0
    r[mask_dem] = np.minimum(1.0, X_cap[mask_dem] / X_dem[mask_dem])

    # 3. Bottleneck constraints per using sector j: s_j = min_{i: A_ij>0} r_i
    s = np.ones(n, dtype=float)
    for j in range(n):
        suppliers = A[:, j] > 0.0
        if suppliers.any():
            s[j] = r[suppliers].min()
        else:
            s[j] = 1.0

    # 4. Constrained intermediate flows Z_con (row-wise scaling)
    #    Z_con[i,:] = Z[i,:] * min(s_i, 1 - sp_i)
    row_factor = np.minimum(s, 1.0 - sp)
    Z_con = row_factor[:, None] * Z

    # 5. Needed flows Z_need given feasible output per user
    #    X_tilde_j = min(X_cap_j, X_dem_j)
    X_tilde = np.minimum(X_cap, X_dem)
    Z_need = A * X_tilde[None, :]  # broadcast over rows i

    # 6. Extra demand E = max(0, Z_need - Z_con)
    E_raw = Z_need - Z_con
    E = np.maximum(E_raw, 0.0)

    # 7. Inventories per producer:
    #    inv_i = max(X_cap_i - FD_eff_i - sum_j Z_con[i,j], 0)
    interm_sales_con = Z_con.sum(axis=1)
    inv = X_cap - FD_eff - interm_sales_con
    inv = np.maximum(inv, 0.0)

    # 8. Aggregate by global sector: inventories & extra demand
    Inv_sec = np.zeros(S, dtype=float)
    Extra_sec = np.zeros(S, dtype=float)
    Extra_sec_j = np.zeros((S, n), dtype=float)

    for i in range(n):
        s_id = globsec_of[i]
        Inv_sec[s_id] += inv[i]
        row_E = E[i, :]
        Extra_sec[s_id] += row_E.sum()
        Extra_sec_j[s_id, :] += row_E

    # 9. Substitution ratios sub_s = min(1, Inv_sec / Extra_sec)
    sub = np.zeros(S, dtype=float)
    mask_sec = Extra_sec > 0.0
    sub[mask_sec] = np.minimum(1.0, Inv_sec[mask_sec] / Extra_sec[mask_sec])

    # 10. γ-based inventories reallocation within each global sector
    Z_new = Z_con.copy()

    for s_id in range(S):
        if Extra_sec[s_id] <= 0.0 or Inv_sec[s_id] <= 0.0:
            continue

        frac = gamma * sub[s_id]
        if frac <= 0.0:
            continue

        i_idx = np.where(globsec_of == s_id)[0]
        inv_i = inv[i_idx]
        total_inv_i = inv_i.sum()
        if total_inv_i <= 0.0:
            continue

        inv_share = inv_i / total_inv_i  # shape (len(i_idx),)

        extra_s_j = Extra_sec_j[s_id, :]  # shape (n,)
        delivered_s_j = frac * extra_s_j  # shape (n,)

        # ΔZ_block[i_rel, j] = inv_share[i_rel] * delivered_s_j[j]
        delta_Z_block = np.outer(inv_share, delivered_s_j)  # (len(i_idx), n)
        Z_new[i_idx, :] += delta_Z_block

    # 11. Cap by needed flows: Z_new <= Z_need
    Z_new = np.minimum(Z_new, Z_need)

    # 12. New gross output from accounting: X_new = FD_eff + sum_j Z_new[i,j]
    X_new = FD_eff + Z_new.sum(axis=1)

    aux = {
        "FD_eff": FD_eff,
        "X_cap": X_cap,
        "X_dem": X_dem,
        "r": r,
        "s": s,
        "Z_con": Z_con,
        "Z_need": Z_need,
        "E": E,
        "inv": inv,
        "Inv_sec": Inv_sec,
        "Extra_sec": Extra_sec,
        "sub": sub,
    }

    return Z_new, X_new, aux


def update_demand(
    FD_post: np.ndarray,
    X_new: np.ndarray,
    A: np.ndarray,
    mode: str = "supply_limited",
) -> np.ndarray:
    """
    Update final demand given new output X_new.

    Parameters
    ----------
    FD_post : (n,)
        Baseline post-shock final demand (upper bound).
    X_new : (n,)
        New gross output vector from propagation.
    A : (n, n)
        Technical coefficients matrix.
    mode : {"supply_limited", "fixed"}
        Demand update rule.

    Returns
    -------
    FD_new : (n,)
        Updated final demand vector.
    """
    FD_post = np.asarray(FD_post, dtype=float).reshape(-1)
    X_new = np.asarray(X_new, dtype=float).reshape(-1)

    if mode == "fixed":
        # No change to demand
        return FD_post

    if mode == "supply_limited":
        # From accounting: X = A X + FD  =>  FD = X - A X
        FD_implied = X_new - A @ X_new
        FD_implied = np.maximum(FD_implied, 0.0)
        # Demand cannot exceed the original post-shock demand
        return np.minimum(FD_post, FD_implied)

    raise ValueError(f"Unknown demand update mode: {mode}")
