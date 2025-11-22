# src/io_climate/model.py

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .propagation import propagate_once, update_demand


class IOClimateModel:
    """
    Global input–output climate-risk propagation model with
    simultaneous demand & capacity shocks and within-sector
    trade reallocation (γ).

    Core objects (n = #country–sectors):
        Z : (n, n) intermediate-use matrix (producer i -> user j)
        FD: (n,) final demand vector
        X : (n,) gross output vector
        A : (n, n) technical coefficients matrix
        L : (n, n) Leontief inverse
        globsec_of : (n,) global sector id for each row i
    """

    def __init__(
        self,
        Z: np.ndarray,
        FD: np.ndarray,
        X: np.ndarray,
        globsec_of: np.ndarray,
        A: Optional[np.ndarray] = None,
        L: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        Z : np.ndarray
            Intermediate-use matrix (n x n).
        FD : np.ndarray
            Final demand vector (n,).
        X : np.ndarray
            Gross output vector (n,).
        globsec_of : np.ndarray
            Length-n vector with global-sector id for each country–sector.
        A : np.ndarray, optional
            Technical coefficients matrix A[i,j] = Z[i,j] / X[j].
            If None, it is computed.
        L : np.ndarray, optional
            Leontief inverse (I - A)^(-1). If None, it is computed.
        """
        Z = np.asarray(Z, dtype=float)
        FD = np.asarray(FD, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

        if Z.shape[0] != Z.shape[1]:
            raise ValueError("Z must be a square (n x n) matrix.")
        n = Z.shape[0]

        if FD.shape[0] != n or X.shape[0] != n or globsec_of.shape[0] != n:
            raise ValueError("FD, X, globsec_of must all have length n = Z.shape[0].")

        self.Z = Z
        self.FD = FD
        self.X = X
        self.globsec_of = globsec_of
        self.n = n

        # Technical coefficients
        if A is None:
            A = self._compute_technical_coefficients(Z, X)
        else:
            A = np.asarray(A, dtype=float)
            if A.shape != Z.shape:
                raise ValueError("A must have the same shape as Z.")
        self.A = A

        # Leontief inverse
        if L is None:
            L = self._compute_leontief_inverse(A)
        else:
            L = np.asarray(L, dtype=float)
            if L.shape != Z.shape:
                raise ValueError("L must have the same shape as Z.")
        self.L = L

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        sd: np.ndarray,
        sp: np.ndarray,
        gamma: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        demand_update_mode: str = "supply_limited",
        return_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the IO climate-risk propagation model with iteration.

        Parameters
        ----------
        sd : (n,)
            Demand shocks in [0,1]. sd[j] = fraction of FD_j lost.
        sp : (n,)
            Capacity shocks in [0,1]. sp[i] = fraction of X_i lost.
        gamma : float, default 0.5
            Trade reallocation capacity in [0,1].
        max_iter : int, default 100
            Maximum number of outer iterations.
        tol : float, default 1e-6
            Convergence tolerance (relative change in FD and X).
        demand_update_mode : {"supply_limited", "fixed"}
            How to update final demand across iterations.
        return_history : bool, default False
            If True, returns histories of X and FD.

        Returns
        -------
        results : dict
            Keys:
                "X_final", "FD_final", "Z_final",
                "iterations", "converged", "aux_last",
                and optionally "X_history", "FD_history".
        """
        sd = np.asarray(sd, dtype=float).reshape(-1)
        sp = np.asarray(sp, dtype=float).reshape(-1)

        if sd.shape[0] != self.n or sp.shape[0] != self.n:
            raise ValueError(f"sd and sp must both have length n = {self.n}")

        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0,1].")

        # Post-shock demand and capacity
        FD_post = self.FD * (1.0 - sd)
        X_cap = self.X * (1.0 - sp)

        FD_eff = FD_post.copy()
        X_prev: Optional[np.ndarray] = None

        X_history = []
        FD_history = []

        converged = False

        for it in range(1, max_iter + 1):
            # Single propagation step
            Z_new, X_new, aux = propagate_once(
                Z=self.Z,
                A=self.A,
                L=self.L,
                globsec_of=self.globsec_of,
                FD_eff=FD_eff,
                X_cap=X_cap,
                sp=sp,
                gamma=gamma,
            )

            if return_history:
                X_history.append(X_new.copy())
                FD_history.append(FD_eff.copy())

            # Update demand according to chosen rule
            FD_new = update_demand(
                FD_post=FD_post,
                X_new=X_new,
                A=self.A,
                mode=demand_update_mode,
            )

            # Convergence checks
            fd_norm = np.linalg.norm(FD_eff, ord=1) + 1e-12
            fd_diff = np.linalg.norm(FD_new - FD_eff, ord=1) / fd_norm

            if X_prev is None:
                x_diff = np.inf
            else:
                x_norm = np.linalg.norm(X_prev, ord=1) + 1e-12
                x_diff = np.linalg.norm(X_new - X_prev, ord=1) / x_norm

            if fd_diff < tol and x_diff < tol:
                converged = True
                FD_eff = FD_new
                X_prev = X_new
                break

            # Prepare next iteration
            FD_eff = FD_new
            X_prev = X_new

        results: Dict[str, Any] = {
            "X_final": X_prev if X_prev is not None else X_new,
            "FD_final": FD_eff,
            "Z_final": Z_new,
            "iterations": it,
            "converged": converged,
            "aux_last": aux,
        }

        if return_history:
            results["X_history"] = X_history
            results["FD_history"] = FD_history

        return results

    # ------------------------------------------------------------------ #
    # Internal helper methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_technical_coefficients(
        Z: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute A[i,j] = Z[i,j] / X[j], handling zero outputs safely.
        """
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float).reshape(-1)

        A = np.zeros_like(Z, dtype=float)
        denom = X.copy()
        denom[denom == 0.0] = np.nan
        A = Z / denom[None, :]
        A = np.nan_to_num(A, nan=0.0)
        return A

    @staticmethod
    def _compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
        """
        Compute the Leontief inverse L = (I - A)^(-1).
        """
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        I = np.eye(n)
        try:
            L = np.linalg.inv(I - A)
        except np.linalg.LinAlgError as err:
            raise ValueError("Leontief inverse (I - A)^(-1) is not invertible.") from err
        return L
