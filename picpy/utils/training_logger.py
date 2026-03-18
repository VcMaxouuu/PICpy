"""Progress logger for the PIC model fitting pipeline."""

import numpy as np
from typing import Optional


class TrainingLogger:
    """Structured progress logger for PIC model fitting.

    Prints formatted, consistent output at three verbosity levels.  All
    output goes to stdout via :func:`print`.

    Parameters
    ----------
    verbosity : int, default=0
        Controls the amount of output:

        - ``0`` (silent)   — no output.
        - ``1`` (basic)    — high-level progress: PDB result, per-step
          convergence summary, and final results.
        - ``2`` (detailed) — everything in level 1, plus per-iteration
          objective values inside each FISTA run.

    log_interval : int, default=10
        At ``verbosity=2``, one iteration line is printed every
        *log_interval* iterations.  Has no effect at lower levels.

    Notes
    -----
    The logger is stateful: :meth:`log_path_start` caches step metadata
    so that :meth:`log_convergence` can produce a self-contained summary
    line at ``verbosity=1``.  :meth:`start_refit` sets an internal flag
    used by :meth:`log_iteration` to suppress the active-set count during
    the unpenalized refit phase.

    Examples
    --------
    >>> logger = TrainingLogger(verbosity=1)
    >>> logger.start_pdb_computation(n_simu=5000, alpha=0.05)
    [PDB]   n_simu=5000  alpha=0.050
    >>> logger.log_pdb_result(lambda_value=0.2251)
            lambda = 0.225100
    """

    SILENT = 0
    BASIC = 1
    DETAILED = 2

    # Column widths and indentation
    _INDENT = " " * 8       # continuation lines aligned with tag content
    _STEP_INDENT = " " * 10  # iteration/convergence lines nested in a step
    _TAG_WIDTH = 8           # total width of "[TAG]   " prefix field

    def __init__(self, verbosity: int = 0, log_interval: int = 10):
        if verbosity not in (self.SILENT, self.BASIC, self.DETAILED):
            raise ValueError(
                f"verbosity must be 0, 1, or 2; got {verbosity!r}."
            )
        self.verbosity = verbosity
        self.log_interval = log_interval

        # Internal state
        self._step_cache: Optional[tuple] = None  # (idx, total, lambda, tol)
        self._in_refit: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, line: str, min_level: int = 1) -> None:
        """Print *line* when ``self.verbosity >= min_level``."""
        if self.verbosity >= min_level:
            print(line)

    def _tag(self, tag: str, content: str) -> str:
        """Format a tagged section header: ``'[TAG]   content'``."""
        return f"[{tag}]".ljust(self._TAG_WIDTH) + content

    def _cont(self, content: str) -> str:
        """Format a continuation line indented to match tag content."""
        return self._INDENT + content

    def _step(self, content: str) -> str:
        """Format a line nested inside a path step (deeper indent)."""
        return self._STEP_INDENT + content

    # ------------------------------------------------------------------
    # PDB regularization parameter
    # ------------------------------------------------------------------

    def start_pdb_computation(self, n_simu: int, alpha: float) -> None:
        """Log the start of the PDB null-distribution simulation.

        Parameters
        ----------
        n_simu : int
            Number of Monte Carlo draws.
        alpha : float
            Nominal level for the quantile selector.
        """
        self._emit(self._tag("PDB", f"n_simu={n_simu:,}  alpha={alpha:.3f}"))

    def log_pdb_result(
        self,
        lambda_value: float,
        pdb_statistics: Optional[np.ndarray] = None,
    ) -> None:
        """Log the computed PDB regularization parameter.

        Parameters
        ----------
        lambda_value : float
            Selected regularization parameter.
        pdb_statistics : ndarray of shape (n_simu,) or None
            Null gradient norms from the simulation.  At ``verbosity=2``
            the mean and standard deviation are appended to the output line.
        """
        line = self._cont(f"lambda = {lambda_value:.6f}")
        if pdb_statistics is not None and self.verbosity >= self.DETAILED:
            line += (
                f"  (statistic: mean={np.mean(pdb_statistics):.4f}"
                f"  sd={np.std(pdb_statistics):.4f})"
            )
        self._emit(line)

    # ------------------------------------------------------------------
    # Regularization path
    # ------------------------------------------------------------------

    def start_fitting(
        self,
        n_samples: int,
        n_features: int,
        lambda_reg: float,
        penalty_name: str,
    ) -> None:
        """Log the start of the regularization path.

        Parameters
        ----------
        n_samples : int
            Number of observations.
        n_features : int
            Number of predictors.
        lambda_reg : float
            Target regularization parameter (lambda PDB).
        penalty_name : str
            String representation of the penalty.
        """
        self._emit("")
        self._emit(
            self._tag(
                "Fit",
                f"n={n_samples:,}  p={n_features:,}"
                f"  lambda={lambda_reg:.6f}  penalty={penalty_name}",
            )
        )

    def log_path_start(
        self,
        path_idx: int,
        total_paths: int,
        lambda_current: float,
        rel_tol: float,
    ) -> None:
        """Log the start of one step in the regularization path.

        At ``verbosity=2`` the step header is printed immediately so that
        subsequent iteration lines appear below it.  At ``verbosity=1`` the
        header is held in cache and emitted together with the convergence
        result once :meth:`log_convergence` is called.

        Parameters
        ----------
        path_idx : int
            Zero-based step index.
        total_paths : int
            Total number of steps in the path.
        lambda_current : float
            Regularization parameter for this step.
        rel_tol : float
            Relative convergence tolerance passed to FISTA.
        """
        self._step_cache = (path_idx, total_paths, lambda_current, rel_tol)
        self._in_refit = False

        if self.verbosity >= self.DETAILED:
            self._emit(
                self._cont(
                    f"Step {path_idx + 1}/{total_paths}"
                    f"  lambda={lambda_current:.6f}"
                    f"  tol={rel_tol:.1e}"
                )
            )

    def log_iteration(
        self,
        iteration: int,
        objective: float,
        loss: Optional[float] = None,
        n_active: Optional[int] = None,
    ) -> None:
        """Log one FISTA iteration.

        Only printed when ``verbosity=2`` and ``(iteration + 1)`` is a
        multiple of ``log_interval``.

        Parameters
        ----------
        iteration : int
            Zero-based iteration index.
        objective : float
            Current penalized objective value.
        loss : float or None
            Current unpenalized (smooth) loss.  Omitted during the
            unpenalized refit phase.
        n_active : int or None
            Number of non-zero coefficients.  Omitted during refit.
        """
        if self.verbosity < self.DETAILED:
            return
        if (iteration + 1) % self.log_interval != 0:
            return

        parts = [f"Iter {iteration + 1:5d} | obj={objective:.6f}"]
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if n_active is not None and not self._in_refit:
            parts.append(f"active={n_active:3d}")

        self._emit(self._step("  ".join(parts)), min_level=self.DETAILED)

    def log_convergence(self, n_iter: int) -> None:
        """Log convergence of a FISTA run.

        At ``verbosity=1``, prints a compact one-line step summary using
        the metadata cached by the most recent :meth:`log_path_start` or
        :meth:`start_refit` call.  At ``verbosity=2``, prints an indented
        convergence notice following any iteration lines.

        Parameters
        ----------
        n_iter : int
            Total number of FISTA iterations performed.
        """
        if self.verbosity == self.BASIC:
            if self._in_refit:
                self._emit(
                    self._cont(f"Refit (unpenalized)  converged ({n_iter} iter)")
                )
            elif self._step_cache is not None:
                idx, total, lam, tol = self._step_cache
                self._emit(
                    self._cont(
                        f"Step {idx + 1}/{total}"
                        f"  lambda={lam:.6f}"
                        f"  tol={tol:.1e}"
                        f"  converged ({n_iter} iter)"
                    )
                )
        elif self.verbosity >= self.DETAILED:
            self._emit(self._step(f"converged in {n_iter} iterations"))

    # ------------------------------------------------------------------
    # Refit phase
    # ------------------------------------------------------------------

    def start_refit(self, n_selected: int) -> None:
        """Log the start of the unpenalized refit on the selected support.

        Parameters
        ----------
        n_selected : int
            Number of features in the selected support.
        """
        self._in_refit = True
        if self.verbosity >= self.DETAILED:
            self._emit(
                self._cont(f"Refit (unpenalized)  support={n_selected} features")
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def log_final_results(
        self,
        n_selected: int,
        objective_final: float,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Log the final model summary after fitting completes.

        Parameters
        ----------
        n_selected : int
            Number of selected features (non-zero coefficients).
        objective_final : float
            Final penalized objective value.
        extra_metrics : dict or None
            Optional additional metrics to display (e.g. training score).
            Keys are metric names; values are floats.
        """
        self._emit("")
        summary = f"selected={n_selected}   objective={objective_final:.6f}"
        self._emit(self._tag("Done", summary))
        if extra_metrics:
            for name, value in extra_metrics.items():
                self._emit(self._cont(f"{name} = {value:.6f}"))
