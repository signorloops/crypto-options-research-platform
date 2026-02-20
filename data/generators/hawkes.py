"""
Hawkes process simulator for modeling self-exciting trade arrivals.

Hawkes process is a point process where past events increase the probability
of future events - perfect for modeling trading clustering in financial markets.

Reference:
- Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes.
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class HawkesParameters:
    """Parameters for Hawkes process.

    The intensity function is:
        λ(t) = μ + Σ α·exp(-β(t - t_i))

    where:
        μ: baseline intensity (background arrival rate)
        α: excitation factor (how much one event increases future intensity)
        β: decay rate (how fast the excitation effect disappears)

    For a stable process, we need: α < β (branching ratio < 1)
    """
    mu: float = 0.1      # Baseline intensity (events per second)
    alpha: float = 0.5   # Excitation magnitude
    beta: float = 1.0    # Decay rate

    def __post_init__(self):
        """Validate parameters."""
        if self.mu <= 0:
            raise ValueError("mu (baseline intensity) must be positive")
        if self.alpha < 0:
            raise ValueError("alpha (excitation) must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta (decay rate) must be positive")
        if self.alpha >= self.beta:
            raise ValueError("For stability, need alpha < beta (branching ratio < 1)")

    @property
    def branching_ratio(self) -> float:
        """Expected number of offspring events per parent event."""
        return self.alpha / self.beta

    @property
    def long_term_intensity(self) -> float:
        """Stationary intensity: λ* = μ / (1 - α/β)"""
        return self.mu / (1 - self.branching_ratio)


class HawkesProcess:
    """Hawkes process simulator using Ogata's thinning algorithm."""

    def __init__(self, params: Optional[HawkesParameters] = None):
        """Initialize Hawkes process.

        Args:
            params: HawkesParameters instance. If None, uses default parameters.
        """
        self.params = params or HawkesParameters()
        self.events: List[float] = []

    def intensity(self, t: float, events: List[float]) -> float:
        """Compute conditional intensity λ(t) given history.

        λ(t) = μ + Σ α·exp(-β(t - t_i)) for t_i < t
        """
        intensity = self.params.mu

        for t_i in events:
            if t_i < t:
                intensity += self.params.alpha * np.exp(
                    -self.params.beta * (t - t_i)
                )

        return intensity

    def simulate(self, T: float, seed: Optional[int] = None) -> List[float]:
        """Simulate Hawkes process using Ogata's thinning algorithm.

        Uses O(1) recursive kernel update instead of O(n) full summation:
            A(n) = exp(-beta * (t_n - t_{n-1})) * (A(n-1) + alpha)
            lambda(t) = mu + A(n)

        Args:
            T: Time horizon (in seconds)
            seed: Random seed for reproducibility

        Returns:
            List of event times
        """
        if seed is not None:
            np.random.seed(seed)

        events = []
        t = 0.0
        A = 0.0  # recursive kernel accumulator
        t_last = 0.0

        lambda_max = self.params.mu

        while t < T:
            u = np.random.exponential(1.0 / lambda_max)
            t = t + u

            if t >= T:
                break

            # O(1) intensity via recursive update
            A = np.exp(-self.params.beta * (t - t_last)) * A
            lambda_t = self.params.mu + A

            if np.random.uniform() <= lambda_t / lambda_max:
                events.append(t)
                A += self.params.alpha
                lambda_max = lambda_t + self.params.alpha
                t_last = t

        self.events = events
        return events

    def simulate_with_mark(self, T: float, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """Simulate Hawkes process with trade sizes (marks).

        Trade sizes are generated from log-normal distribution.

        Returns:
            Tuple of (event_times, trade_sizes)
        """
        times = self.simulate(T, seed)

        # Generate trade sizes (log-normal: typical for financial data)
        # Mean = 1.0, std = 0.5 (in log space)
        sizes = np.random.lognormal(mean=0.0, sigma=0.5, size=len(times))

        return times, sizes.tolist()

    def get_event_counts(self, window_size: float, T: float) -> np.ndarray:
        """Count events in time windows.

        Args:
            window_size: Size of each window (in seconds)
            T: Total time horizon

        Returns:
            Array of event counts per window
        """
        if not self.events:
            return np.array([])

        n_windows = int(T / window_size)
        counts = np.zeros(n_windows)

        for t in self.events:
            if t < T:
                idx = int(t / window_size)
                if idx < n_windows:
                    counts[idx] += 1

        return counts

    def get_inter_event_times(self) -> np.ndarray:
        """Compute inter-event times (waiting times between consecutive events)."""
        if len(self.events) < 2:
            return np.array([])
        return np.diff(self.events)


class PoissonProcess:
    """Simple Poisson process for comparison with Hawkes process."""

    def __init__(self, intensity: float = 0.1):
        """Initialize Poisson process.

        Args:
            intensity: Constant arrival rate (events per second)
        """
        if intensity <= 0:
            raise ValueError("Intensity must be positive")
        self.intensity = intensity
        self.events: List[float] = []

    def simulate(self, T: float, seed: Optional[int] = None) -> List[float]:
        """Simulate Poisson process.

        Args:
            T: Time horizon (in seconds)
            seed: Random seed

        Returns:
            List of event times
        """
        if seed is not None:
            np.random.seed(seed)

        # Number of events ~ Poisson(λT)
        n_events = np.random.poisson(self.intensity * T)

        # Event times ~ Uniform(0, T)
        events = np.sort(np.random.uniform(0, T, n_events)).tolist()

        self.events = events
        return events

    def get_event_counts(self, window_size: float, T: float) -> np.ndarray:
        """Count events in time windows."""
        if not self.events:
            return np.array([])

        n_windows = int(T / window_size)
        counts = np.zeros(n_windows)

        for t in self.events:
            if t < T:
                idx = int(t / window_size)
                if idx < n_windows:
                    counts[idx] += 1

        return counts

    def get_inter_event_times(self) -> np.ndarray:
        """Compute inter-event times."""
        if len(self.events) < 2:
            return np.array([])
        return np.diff(self.events)


def estimate_hawkes_params(event_times: List[float]) -> HawkesParameters:
    """Estimate Hawkes parameters from event data using moment matching.

    This is a simple estimation method. For more accurate estimation,
    use maximum likelihood estimation (MLE).

    Args:
        event_times: List of observed event times

    Returns:
        Estimated HawkesParameters
    """
    if len(event_times) < 10:
        raise ValueError("Need at least 10 events for estimation")

    T = max(event_times)
    n_events = len(event_times)

    # Estimate baseline intensity
    mu_est = n_events / T * 0.5  # Rough estimate

    # Estimate clustering from inter-event times
    iet = np.diff(sorted(event_times))
    cv = np.std(iet) / np.mean(iet)  # Coefficient of variation

    # Higher CV suggests more clustering (higher alpha)
    alpha_est = min(0.8, max(0.1, cv - 1.0))
    beta_est = alpha_est + 0.5  # Ensure stability

    return HawkesParameters(mu=mu_est, alpha=alpha_est, beta=beta_est)
