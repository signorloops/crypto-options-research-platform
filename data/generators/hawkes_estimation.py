"""
Hawkes process parameter estimation from real trade data.

Methods implemented:
1. Maximum Likelihood Estimation (MLE) - most accurate
2. Method of Moments (MoM) - fast approximation
3. Least Squares Estimation - alternative approach

References:
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes.
- Veen, A., & Schoenberg, F.P. (2008). Estimation of space-time branching process models.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import expi
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

from data.generators.hawkes import HawkesParameters


@dataclass
class EstimationResult:
    """Result of parameter estimation."""
    params: HawkesParameters
    log_likelihood: float
    method: str
    success: bool
    message: str
    n_events: int
    computation_time: float

    def __str__(self) -> str:
        return (f"HawkesParameterEstimation(\n"
                f"  params={self.params},\n"
                f"  log_likelihood={self.log_likelihood:.2f},\n"
                f"  method='{self.method}',\n"
                f"  success={self.success},\n"
                f"  n_events={self.n_events}\n"
                f")")


class HawkesEstimator:
    """Estimate Hawkes process parameters from event data."""

    def __init__(self, event_times: List[float]):
        """
        Initialize estimator with event data.

        Args:
            event_times: List of event timestamps (sorted)
        """
        self.event_times = np.sort(event_times)
        self.n_events = len(event_times)
        self.T = self.event_times[-1] - self.event_times[0] if len(event_times) > 0 else 0

        if self.n_events < 10:
            raise ValueError("Need at least 10 events for reliable estimation")

    def log_likelihood(self, mu: float, alpha: float, beta: float) -> float:
        """
        Compute log-likelihood of Hawkes process.

        For Hawkes process with exponential kernel:
        λ(t) = μ + Σ α·exp(-β(t - t_i))

        Log-likelihood = Σ log(λ(t_i)) - ∫λ(t)dt

        Args:
            mu: Baseline intensity
            alpha: Excitation magnitude
            beta: Decay rate

        Returns:
            Log-likelihood value
        """
        if mu <= 0 or alpha < 0 or beta <= 0:
            return -np.inf

        if alpha >= beta:  # Unstable process
            return -np.inf

        # Compute intensity at each event time
        log_sum = 0.0

        for i in range(self.n_events):
            t_i = self.event_times[i]

            # λ(t_i) = μ + Σ_{t_j < t_i} α·exp(-β(t_i - t_j))
            intensity = mu

            for j in range(i):
                t_j = self.event_times[j]
                dt = t_i - t_j
                intensity += alpha * np.exp(-beta * dt)

            if intensity <= 0:
                return -np.inf

            log_sum += np.log(intensity)

        # Compensator term: ∫λ(t)dt
        # For Hawkes: ∫λ(t)dt = μ·T + (α/β)·Σ[1 - exp(-β(T - t_i))]
        T_end = self.event_times[-1]
        T_start = self.event_times[0]
        time_span = T_end - T_start

        compensator = mu * time_span

        for t_i in self.event_times:
            compensator += (alpha / beta) * (1 - np.exp(-beta * (T_end - t_i)))

        return log_sum - compensator

    def estimate_mle(self,
                     mu_init: Optional[float] = None,
                     alpha_init: Optional[float] = None,
                     beta_init: Optional[float] = None) -> EstimationResult:
        """
        Estimate parameters using Maximum Likelihood Estimation.

        This is the most accurate method but computationally intensive.

        Returns:
            EstimationResult with estimated parameters
        """
        import time
        start_time = time.time()

        # Initial guesses
        if mu_init is None:
            mu_init = self.n_events / self.T * 0.5
        if alpha_init is None:
            alpha_init = 0.3
        if beta_init is None:
            beta_init = 1.0

        x0 = np.array([mu_init, alpha_init, beta_init])

        # Objective function (negative log-likelihood)
        def objective(x):
            mu, alpha, beta = x
            return -self.log_likelihood(mu, alpha, beta)

        # Bounds: mu > 0, alpha >= 0, beta > 0
        bounds = [(1e-6, None), (0, None), (1e-6, None)]

        # Constraint: alpha < beta (for stability)
        def stability_constraint(x):
            mu, alpha, beta = x
            return beta - alpha - 1e-6  # Must be > 0

        constraints = {'type': 'ineq', 'fun': stability_constraint}

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            computation_time = time.time() - start_time

            if result.success:
                mu_est, alpha_est, beta_est = result.x
                params = HawkesParameters(
                    mu=float(mu_est),
                    alpha=float(alpha_est),
                    beta=float(beta_est)
                )

                return EstimationResult(
                    params=params,
                    log_likelihood=-result.fun,
                    method='MLE (SLSQP)',
                    success=True,
                    message='Optimization converged',
                    n_events=self.n_events,
                    computation_time=computation_time
                )
            else:
                # Fall back to differential evolution
                return self._estimate_mle_global()

        except Exception as e:
            return self._estimate_mle_global()

    def _estimate_mle_global(self) -> EstimationResult:
        """
        Use global optimization (differential evolution) as fallback.
        More robust but slower.
        """
        import time
        start_time = time.time()

        # Define bounds based on data characteristics
        n_events = self.n_events
        T = self.T

        # Reasonable bounds
        mu_max = n_events / T * 2  # Can't be more than 2x observed rate
        bounds = [
            (1e-6, mu_max),      # mu
            (0, 0.9),            # alpha (must be < beta, so max 0.9)
            (1e-3, 10.0)         # beta
        ]

        def objective(x):
            mu, alpha, beta = x
            if alpha >= beta:
                return 1e10  # Invalid region
            return -self.log_likelihood(mu, alpha, beta)

        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            seed=42,
            workers=-1,  # Parallel processing
            polish=True
        )

        computation_time = time.time() - start_time

        mu_est, alpha_est, beta_est = result.x
        params = HawkesParameters(
            mu=float(mu_est),
            alpha=float(alpha_est),
            beta=float(beta_est)
        )

        return EstimationResult(
            params=params,
            log_likelihood=-result.fun,
            method='MLE (Global)',
            success=result.success,
            message='Global optimization completed',
            n_events=self.n_events,
            computation_time=computation_time
        )

    def estimate_moments(self) -> EstimationResult:
        """
        Estimate using Method of Moments (fast approximation).

        Based on:
        - Mean cluster size = 1 / (1 - α/β)
        - Variance of counts in windows

        Returns:
            EstimationResult
        """
        import time
        start_time = time.time()

        # Divide into windows
        n_windows = min(50, self.n_events // 5)
        window_size = self.T / n_windows

        counts = np.zeros(n_windows)
        for t in self.event_times:
            idx = min(int((t - self.event_times[0]) / window_size), n_windows - 1)
            counts[idx] += 1

        # Method of moments estimation
        mean_count = np.mean(counts)
        var_count = np.var(counts)

        if var_count <= mean_count:
            # No overdispersion - fallback to Poisson-like
            mu_est = mean_count / window_size
            alpha_est = 0.01
            beta_est = 1.0
        else:
            # Overdispersion suggests Hawkes process
            # Var/Mean ratio relates to branching ratio
            ratio = var_count / mean_count
            branching_ratio = 1 - 1 / ratio if ratio > 1 else 0.3
            branching_ratio = np.clip(branching_ratio, 0.01, 0.9)

            # Estimate parameters
            mu_est = (mean_count / window_size) * (1 - branching_ratio)
            beta_est = 2.0  # Default decay
            alpha_est = branching_ratio * beta_est

        computation_time = time.time() - start_time

        try:
            params = HawkesParameters(
                mu=float(mu_est),
                alpha=float(alpha_est),
                beta=float(beta_est)
            )

            ll = self.log_likelihood(mu_est, alpha_est, beta_est)

            return EstimationResult(
                params=params,
                log_likelihood=ll,
                method='Method of Moments',
                success=True,
                message='Moment matching completed',
                n_events=self.n_events,
                computation_time=computation_time
            )
        except ValueError as e:
            # Invalid parameters - return fallback
            return EstimationResult(
                params=HawkesParameters(mu=0.1, alpha=0.1, beta=1.0),
                log_likelihood=-np.inf,
                method='Method of Moments (Fallback)',
                success=False,
                message=str(e),
                n_events=self.n_events,
                computation_time=computation_time
            )

    def estimate_ls(self) -> EstimationResult:
        """
        Least Squares estimation using autocorrelation.

        Fits the theoretical autocorrelation to empirical data.

        Returns:
            EstimationResult
        """
        import time
        start_time = time.time()

        # Compute empirical autocorrelation of counts
        n_windows = min(100, self.n_events // 3)
        window_size = self.T / n_windows

        counts = np.zeros(n_windows)
        for t in self.event_times:
            idx = min(int((t - self.event_times[0]) / window_size), n_windows - 1)
            counts[idx] += 1

        # Compute autocorrelation
        counts_centered = counts - np.mean(counts)
        autocorr = np.correlate(counts_centered, counts_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only non-negative lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Fit exponential decay: autocorr(k) ~ exp(-(β-α)·k·Δt)
        if len(autocorr) < 3 or autocorr[1] <= 0:
            # Not enough data or no correlation
            return self.estimate_moments()

        # Estimate decay rate from first few lags
        decay_rate = -np.log(autocorr[1]) / window_size if autocorr[1] > 0 else 1.0

        # Branching ratio from variance
        var = np.var(counts)
        mean = np.mean(counts)
        branching_ratio = 1 - mean / var if var > mean else 0.3
        branching_ratio = np.clip(branching_ratio, 0.01, 0.9)

        # Solve for parameters
        # decay_rate ≈ β - α
        # branching_ratio = α / β
        beta_est = decay_rate / (1 - branching_ratio) if branching_ratio < 1 else decay_rate
        alpha_est = branching_ratio * beta_est

        # Baseline from mean
        mu_est = (mean / window_size) * (1 - branching_ratio)

        computation_time = time.time() - start_time

        try:
            params = HawkesParameters(
                mu=float(mu_est),
                alpha=float(alpha_est),
                beta=float(beta_est)
            )

            ll = self.log_likelihood(mu_est, alpha_est, beta_est)

            return EstimationResult(
                params=params,
                log_likelihood=ll,
                method='Least Squares',
                success=True,
                message='LS fitting completed',
                n_events=self.n_events,
                computation_time=computation_time
            )
        except ValueError:
            return self.estimate_moments()

    def compare_methods(self) -> Dict[str, EstimationResult]:
        """
        Run all estimation methods and compare results.

        Returns:
            Dictionary of method name -> EstimationResult
        """
        results = {}

        # Method of Moments (fastest)
        print("Running Method of Moments...")
        results['MoM'] = self.estimate_moments()

        # Least Squares
        print("Running Least Squares...")
        results['LS'] = self.estimate_ls()

        # MLE (most accurate, but slow)
        print("Running Maximum Likelihood Estimation...")
        results['MLE'] = self.estimate_mle()

        return results


def load_trades_from_csv(filepath: str, timestamp_col: str = 'timestamp') -> List[float]:
    """
    Load trade timestamps from CSV file.

    Args:
        filepath: Path to CSV file
        timestamp_col: Name of timestamp column

    Returns:
        List of trade timestamps in seconds
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    # Convert timestamps to seconds (relative to first trade)
    timestamps = pd.to_datetime(df[timestamp_col])
    seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().tolist()

    return seconds


def analyze_trade_clustering(trade_times: List[float],
                            window_seconds: float = 60) -> Dict:
    """
    Analyze real trade data for Hawkes-like clustering.

    Args:
        trade_times: List of trade timestamps
        window_seconds: Window size for analysis

    Returns:
        Dictionary with analysis results
    """
    trade_times = np.sort(trade_times)
    n_trades = len(trade_times)
    T = trade_times[-1] - trade_times[0]

    # Basic statistics
    iet = np.diff(trade_times)

    stats = {
        'n_trades': n_trades,
        'duration_seconds': T,
        'trades_per_minute': n_trades / (T / 60),
        'mean_inter_event': np.mean(iet),
        'std_inter_event': np.std(iet),
        'cv': np.std(iet) / np.mean(iet),  # Coefficient of variation
        'min_iet': np.min(iet),
        'max_iet': np.max(iet),
    }

    # Test for clustering
    # Poisson: CV ≈ 1
    # Hawkes: CV > 1 (overdispersion)
    if stats['cv'] > 1.5:
        stats['clustering_assessment'] = 'Strong clustering - Hawkes model recommended'
    elif stats['cv'] > 1.2:
        stats['clustering_assessment'] = 'Moderate clustering - Hawkes may help'
    elif stats['cv'] > 1.05:
        stats['clustering_assessment'] = 'Weak clustering'
    else:
        stats['clustering_assessment'] = 'No significant clustering - Poisson may suffice'

    # Estimate parameters
    try:
        estimator = HawkesEstimator(trade_times)
        result = estimator.estimate_moments()  # Fast method
        stats['estimated_params'] = {
            'mu': result.params.mu,
            'alpha': result.params.alpha,
            'beta': result.params.beta,
            'branching_ratio': result.params.branching_ratio
        }
    except Exception as e:
        stats['estimated_params'] = None
        stats['estimation_error'] = str(e)

    return stats


if __name__ == '__main__':
    # Demo: Estimate from synthetic data
    from data.generators.hawkes import HawkesProcess

    print("=" * 60)
    print("Hawkes Parameter Estimation Demo")
    print("=" * 60)

    # Generate synthetic data with known parameters
    true_params = HawkesParameters(mu=0.1, alpha=0.5, beta=1.0)
    hawkes = HawkesProcess(true_params)
    events = hawkes.simulate(T=3600, seed=42)  # 1 hour

    print(f"\nTrue parameters:")
    print(f"  μ = {true_params.mu}, α = {true_params.alpha}, β = {true_params.beta}")
    print(f"  Generated {len(events)} events")

    # Estimate parameters
    estimator = HawkesEstimator(events)
    results = estimator.compare_methods()

    print("\n" + "=" * 60)
    print("Estimation Results:")
    print("=" * 60)

    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Parameters: μ={result.params.mu:.4f}, "
              f"α={result.params.alpha:.4f}, β={result.params.beta:.4f}")
        print(f"  Branching ratio: {result.params.branching_ratio:.4f}")
        print(f"  Log-likelihood: {result.log_likelihood:.2f}")
        print(f"  Time: {result.computation_time:.3f}s")
        print(f"  Success: {result.success}")
