"""
Online Bayesian Changepoint Detection (BOCD).

Reference:
- Adams, R. P., & MacKay, D. J. C. (2007)
"""
from dataclasses import dataclass
from math import lgamma, log, pi
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Student-t predictive log-density under Normal-Inverse-Gamma posterior."""
    dof = max(2.0 * alpha, 1e-8)
    scale2 = beta * (kappa + 1.0) / max(alpha * kappa, 1e-12)
    scale2 = max(scale2, 1e-12)
    z2 = ((x - mu) ** 2) / scale2
    return (
        lgamma((dof + 1.0) / 2.0)
        - lgamma(dof / 2.0)
        - 0.5 * (log(dof) + log(pi) + log(scale2))
        - ((dof + 1.0) / 2.0) * log(1.0 + z2 / dof)
    )


@dataclass
class BOCDConfig:
    """Configuration for BOCD."""
    hazard_lambda: float = 250.0
    max_run_length: int = 500
    prior_mean: float = 0.0
    prior_kappa: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    alert_threshold: float = 0.5


class OnlineBayesianChangepointDetector:
    """
    Online Bayesian changepoint detector for streaming returns.

    Uses conjugate Gaussian model with unknown mean/variance.
    """

    def __init__(self, config: Optional[BOCDConfig] = None):
        config = config or BOCDConfig()
        if config.hazard_lambda <= 0:
            raise ValueError("hazard_lambda must be positive")
        if config.max_run_length <= 1:
            raise ValueError("max_run_length must be > 1")
        self.config = config
        self.reset()

    def reset(self) -> None:
        m = self.config.max_run_length
        self.run_length_probs = np.zeros(m + 1, dtype=float)
        self.run_length_probs[0] = 1.0

        self.mu = np.full(m + 1, self.config.prior_mean, dtype=float)
        self.kappa = np.full(m + 1, self.config.prior_kappa, dtype=float)
        self.alpha = np.full(m + 1, self.config.prior_alpha, dtype=float)
        self.beta = np.full(m + 1, self.config.prior_beta, dtype=float)

        self.changepoint_probabilities = []
        self.run_length_mode = []
        self.observation_count = 0

    def _hazard(self) -> float:
        return 1.0 / self.config.hazard_lambda

    @staticmethod
    def _posterior_update(x: float, mu: float, kappa: float, alpha: float, beta: float):
        """Conjugate posterior update for Normal-Inverse-Gamma."""
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa_new
        return mu_new, kappa_new, alpha_new, beta_new

    def update(self, x: float) -> float:
        """Consume one observation and return changepoint probability."""
        m = self.config.max_run_length
        hazard = self._hazard()
        x = float(x)

        pred_logp = np.empty(m + 1, dtype=float)
        for r in range(m + 1):
            pred_logp[r] = _student_t_logpdf(x, self.mu[r], self.kappa[r], self.alpha[r], self.beta[r])
        pred_p = np.exp(pred_logp - np.max(pred_logp))
        pred_cp = np.exp(
            _student_t_logpdf(
                x,
                self.config.prior_mean,
                self.config.prior_kappa,
                self.config.prior_alpha,
                self.config.prior_beta,
            )
            - np.max(pred_logp)
        )

        growth = self.run_length_probs * pred_p * (1.0 - hazard)
        cp = float(np.sum(self.run_length_probs * hazard) * pred_cp)

        new_probs = np.zeros_like(self.run_length_probs)
        new_probs[0] = cp
        new_probs[1:] = growth[:-1]
        norm = float(new_probs.sum())
        if norm <= 0:
            new_probs[:] = 0.0
            new_probs[0] = 1.0
        else:
            new_probs /= norm

        new_mu = np.full(m + 1, self.config.prior_mean, dtype=float)
        new_kappa = np.full(m + 1, self.config.prior_kappa, dtype=float)
        new_alpha = np.full(m + 1, self.config.prior_alpha, dtype=float)
        new_beta = np.full(m + 1, self.config.prior_beta, dtype=float)

        # Grow existing run-length posteriors.
        for r in range(m):
            mu_r, kappa_r, alpha_r, beta_r = self._posterior_update(
                x, self.mu[r], self.kappa[r], self.alpha[r], self.beta[r]
            )
            new_mu[r + 1] = mu_r
            new_kappa[r + 1] = kappa_r
            new_alpha[r + 1] = alpha_r
            new_beta[r + 1] = beta_r

        # Run length 0 corresponds to a changepoint then one new sample.
        mu0, kappa0, alpha0, beta0 = self._posterior_update(
            x,
            self.config.prior_mean,
            self.config.prior_kappa,
            self.config.prior_alpha,
            self.config.prior_beta,
        )
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0

        self.run_length_probs = new_probs
        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta
        self.observation_count += 1

        cp_prob = float(self.run_length_probs[0])
        self.changepoint_probabilities.append(cp_prob)
        self.run_length_mode.append(int(np.argmax(self.run_length_probs)))
        return cp_prob

    def is_changepoint(self, x: float, threshold: float = None) -> bool:
        """Update detector and return changepoint alarm."""
        cp_prob = self.update(x)
        th = self.config.alert_threshold if threshold is None else threshold
        return cp_prob >= th

    def get_state(self) -> Dict[str, float]:
        """Return summary state."""
        cp_prob = float(self.changepoint_probabilities[-1]) if self.changepoint_probabilities else 0.0
        run_mode = int(self.run_length_mode[-1]) if self.run_length_mode else 0
        expected_run = float(np.dot(np.arange(len(self.run_length_probs)), self.run_length_probs))
        return {
            "observation_count": float(self.observation_count),
            "changepoint_probability": cp_prob,
            "run_length_mode": float(run_mode),
            "expected_run_length": expected_run,
        }

    def update_batch(self, values: Sequence[float]) -> np.ndarray:
        """Consume multiple observations and return changepoint probabilities."""
        out = np.zeros(len(values), dtype=float)
        for i, value in enumerate(values):
            out[i] = self.update(float(value))
        return out

    def top_changepoints(
        self,
        top_k: int = 5,
        min_probability: float = 0.1,
    ) -> List[Tuple[int, float]]:
        """
        Return top-k changepoint candidates as (index, probability).
        """
        if top_k <= 0:
            return []
        if not self.changepoint_probabilities:
            return []
        cp = np.asarray(self.changepoint_probabilities, dtype=float)
        idx = np.where(cp >= min_probability)[0]
        if len(idx) == 0:
            return []
        selected = sorted(((int(i), float(cp[i])) for i in idx), key=lambda x: x[1], reverse=True)
        return selected[:top_k]
