"""
Volatility Regime Detector using Hidden Markov Models (HMM).

Detects volatility regimes (low/medium/high) for dynamic spread adjustment.
"""

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """Volatility regime states."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class RegimeConfig:
    """Configuration for regime detector."""

    n_regimes: int = 3  # Low, Medium, High volatility
    lookback_window: int = 100  # Window for HMM training
    volatility_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.6])

    # Feature configuration
    use_log_returns: bool = True
    use_realized_vol: bool = True
    rv_windows: List[int] = field(default_factory=lambda: [5, 15, 30])  # minutes
    annualization_periods: float = 365 * 24 * 60  # crypto minute bars (24/7)

    # Online update settings
    min_samples_for_training: int = 50
    retrain_interval: int = 100  # Retrain every N samples

    # Regime switch detection
    switch_probability_threshold: float = 0.7  # Alert if switch prob > 0.7
    regime_persistence_min_samples: int = 3  # Minimum consecutive samples before allowing switch
    min_confidence_for_switch: float = 0.55  # Candidate regime posterior threshold
    switch_hysteresis: float = 0.10  # Candidate posterior must exceed current by this margin

    # Spread adjustment multipliers per regime
    spread_multipliers: Dict[RegimeState, float] = field(
        default_factory=lambda: {
            RegimeState.LOW: 0.8,  # Tighter spreads in low vol
            RegimeState.MEDIUM: 1.0,  # Normal spreads
            RegimeState.HIGH: 1.5,  # Wider spreads in high vol
        }
    )


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""

    log_return: float
    realized_vol_5min: float
    realized_vol_15min: float
    realized_vol_30min: float
    timestamp: datetime


class VolatilityRegimeDetector:
    """
    HMM-based volatility regime detector.

    Uses a Gaussian Hidden Markov Model to detect volatility regimes:
    - Low volatility: Tight spreads, aggressive quoting
    - Medium volatility: Normal spreads
    - High volatility: Wide spreads, conservative quoting

    Features:
    - Log returns
    - Realized volatility (5/15/30 minute windows)
    - Online learning with sliding window
    - Regime switch probability prediction
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model: Optional[hmm.GaussianHMM] = None
        self.current_regime: RegimeState = RegimeState.MEDIUM
        self.regime_probabilities: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])

        # Data buffers
        self._returns_buffer: Deque[float] = deque(maxlen=self.config.lookback_window)
        self._features_buffer: Deque[RegimeFeatures] = deque(maxlen=self.config.lookback_window)
        self._timestamps: Deque[datetime] = deque(maxlen=self.config.lookback_window)

        # Training state
        self._sample_count: int = 0
        self._last_training_sample: int = 0
        self._is_fitted: bool = False
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_scale: Optional[np.ndarray] = None
        self._training_failures: int = 0
        self._last_training_error: str = ""
        self._state_map: Dict[int, int] = {i: i for i in range(self.config.n_regimes)}
        self._current_regime_run_length: int = 0

        # Regime history
        self.regime_history: List[Tuple[datetime, RegimeState, np.ndarray]] = []

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """Initialize HMM model."""
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_regimes, covariance_type="full", n_iter=100, random_state=42
        )

    def _calculate_realized_volatility(self, returns: Sequence[float], window: int) -> float:
        """Calculate realized volatility for given window."""
        if len(returns) < window:
            return float(np.std(returns)) if returns else 0.0

        recent_returns = list(returns)[-window:]
        return float(np.std(recent_returns) * np.sqrt(self.config.annualization_periods))

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize feature vectors using the latest training distribution."""
        if self._feature_mean is None or self._feature_scale is None:
            return X
        return (X - self._feature_mean) / self._feature_scale

    def _extract_features(self, new_return: float) -> RegimeFeatures:
        """Extract features from new return."""
        now = datetime.now(timezone.utc)

        # Update returns buffer
        self._returns_buffer.append(new_return)

        # Calculate realized volatilities
        rv_5 = self._calculate_realized_volatility(self._returns_buffer, 5)
        rv_15 = self._calculate_realized_volatility(self._returns_buffer, 15)
        rv_30 = self._calculate_realized_volatility(self._returns_buffer, 30)

        features = RegimeFeatures(
            log_return=new_return,
            realized_vol_5min=rv_5,
            realized_vol_15min=rv_15,
            realized_vol_30min=rv_30,
            timestamp=now,
        )

        self._features_buffer.append(features)
        self._timestamps.append(now)

        return features

    def _prepare_training_data(self) -> np.ndarray:
        """Prepare feature matrix for HMM training."""
        if len(self._features_buffer) < self.config.min_samples_for_training:
            return np.array([])

        features_list = []
        for f in self._features_buffer:
            feature_vector = [f.log_return]
            if self.config.use_realized_vol:
                feature_vector.extend(
                    [f.realized_vol_5min, f.realized_vol_15min, f.realized_vol_30min]
                )
            features_list.append(feature_vector)

        return np.array(features_list)

    def _train_model(self) -> bool:
        """Train HMM model on buffered data."""
        X = self._prepare_training_data()

        if len(X) < self.config.min_samples_for_training:
            return False

        # Degenerate samples frequently cause unstable covariance estimates.
        # Skip training until we have enough distinct observations.
        if np.unique(X, axis=0).shape[0] < self.config.n_regimes:
            self._training_failures += 1
            self._last_training_error = "insufficient_distinct_samples"
            return False

        try:
            feature_mean = np.mean(X, axis=0)
            feature_scale = np.std(X, axis=0)
            feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
            X_norm = (X - feature_mean) / feature_scale

            # Re-initialize model to avoid state issues
            self._init_model()
            with warnings.catch_warnings():
                try:
                    from sklearn.exceptions import ConvergenceWarning

                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                except Exception:
                    warnings.filterwarnings("ignore", message=".*converge.*")
                self.model.fit(X_norm)

            # Check if model converged properly
            if hasattr(self.model, "transmat_"):
                # Fix any rows that sum to 0 (no transitions observed)
                transmat = self.model.transmat_
                for i in range(len(transmat)):
                    if transmat[i].sum() == 0:
                        # Set uniform probabilities for this row
                        transmat[i] = np.ones(self.config.n_regimes) / self.config.n_regimes
                    # Normalize to ensure rows sum to 1
                    transmat[i] = transmat[i] / transmat[i].sum()
                self.model.transmat_ = transmat

            # Map states by conditional variance/risk, not mean returns.
            self._state_map = self._build_state_map_from_model()

            self._is_fitted = True
            self._last_training_sample = self._sample_count
            self._feature_mean = feature_mean
            self._feature_scale = feature_scale
            self._training_failures = 0
            self._last_training_error = ""
            return True
        except Exception as e:
            # Model fitting failed, will retry later
            self._training_failures += 1
            self._last_training_error = str(e)
            if self._training_failures <= 3 or self._training_failures % 50 == 0:
                logger.warning(
                    "Regime detector training failed",
                    extra={
                        "failures": self._training_failures,
                        "error": self._last_training_error,
                    },
                )
            return False

    def _build_state_map_from_model(self) -> Dict[int, int]:
        """Build mapping old_state -> ordered_state(LOW..HIGH) by conditional variance."""
        if self.model is None:
            return {i: i for i in range(self.config.n_regimes)}

        risk_scores: List[float] = []
        covars = getattr(self.model, "covars_", None)
        if covars is not None and len(covars) == self.config.n_regimes:
            for i in range(self.config.n_regimes):
                cov = np.asarray(covars[i], dtype=float)
                if cov.ndim == 0:
                    score = float(abs(cov))
                elif cov.ndim == 1:
                    score = float(np.mean(np.abs(cov)))
                else:
                    # Average diagonal variance as state risk score.
                    score = float(np.trace(cov) / max(cov.shape[0], 1))
                risk_scores.append(score)
        else:
            means = getattr(self.model, "means_", np.zeros((self.config.n_regimes, 1)))
            risk_scores = [float(abs(np.asarray(means[i])[0])) for i in range(self.config.n_regimes)]

        order = np.argsort(np.asarray(risk_scores, dtype=float))
        return {int(old): int(new) for new, old in enumerate(order)}

    def _apply_sticky_transition(self, candidate: RegimeState, probs: np.ndarray) -> RegimeState:
        """Apply persistence and hysteresis to reduce regime flip noise."""
        if candidate == self.current_regime:
            return candidate

        current_idx = int(self.current_regime.value)
        candidate_idx = int(candidate.value)
        if candidate_idx >= len(probs) or current_idx >= len(probs):
            return self.current_regime

        candidate_prob = float(probs[candidate_idx])
        current_prob = float(probs[current_idx])
        if self._current_regime_run_length < self.config.regime_persistence_min_samples:
            return self.current_regime
        if candidate_prob < self.config.min_confidence_for_switch:
            return self.current_regime
        if candidate_prob - current_prob < self.config.switch_hysteresis:
            return self.current_regime
        return candidate

    def _predict_regime(self) -> Tuple[RegimeState, np.ndarray]:
        """Predict current regime using HMM."""
        if not self._is_fitted or len(self._features_buffer) == 0:
            return RegimeState.MEDIUM, np.array([1 / 3, 1 / 3, 1 / 3])

        try:
            # Get latest features
            latest = self._features_buffer[-1]
            feature_vector = [latest.log_return]
            if self.config.use_realized_vol:
                feature_vector.extend(
                    [latest.realized_vol_5min, latest.realized_vol_15min, latest.realized_vol_30min]
                )

            X = np.array([feature_vector], dtype=float)
            X = self._normalize_features(X)

            # Predict regime
            hidden_state = self.model.predict(X)[0]
            _, posteriors = self.model.score_samples(X)

            # Get regime probabilities and remap to sorted state order
            raw_probs = posteriors[0]
            mapped_state = self._state_map.get(int(hidden_state), int(hidden_state))
            regime_probs = np.zeros_like(raw_probs)
            for old_idx, new_idx in self._state_map.items():
                regime_probs[new_idx] = raw_probs[old_idx]

            regime = RegimeState(mapped_state)

            return regime, regime_probs
        except (ValueError, np.linalg.LinAlgError):
            # HMM prediction failed (e.g., covariance not positive definite)
            # Fall back to current regime or MEDIUM
            return self.current_regime, self.regime_probabilities

    def update(self, returns: float) -> RegimeState:
        """
        Update detector with new return and return current regime.

        Args:
            returns: Log return (or simple return if use_log_returns=False)

        Returns:
            Current regime state
        """
        self._sample_count += 1

        # Convert to log return if needed
        log_return = np.log1p(returns) if self.config.use_log_returns and returns > -1 else returns

        # Extract features
        self._extract_features(log_return)

        # Train model if needed
        if (
            not self._is_fitted
            or self._sample_count - self._last_training_sample >= self.config.retrain_interval
        ):
            self._train_model()

        # Predict regime if model is fitted
        if self._is_fitted:
            regime, probs = self._predict_regime()
            selected_regime = self._apply_sticky_transition(regime, probs)
            if selected_regime == self.current_regime:
                self._current_regime_run_length += 1
            else:
                self.current_regime = selected_regime
                self._current_regime_run_length = 1
            self.regime_probabilities = probs

            # Record history
            self.regime_history.append((datetime.now(timezone.utc), self.current_regime, probs.copy()))

        return self.current_regime

    def predict_regime_switch_probability(self) -> float:
        """
        Predict probability of regime switch in next period.

        Returns:
            Probability of regime switch (0-1)
        """
        if not self._is_fitted or len(self.regime_history) < 2:
            return 0.0

        # Get transition matrix
        transmat = self.model.transmat_

        # Current regime index
        current_idx = self.current_regime.value

        # Probability of staying in current regime
        stay_prob = transmat[current_idx, current_idx]

        # Probability of switching
        switch_prob = 1.0 - stay_prob

        return switch_prob

    def get_regime_switch_alert(self) -> Tuple[bool, str]:
        """
        Check if regime switch alert should be triggered.

        Returns:
            Tuple of (alert_triggered, message)
        """
        if not self._is_fitted:
            return False, "Model not trained yet"

        switch_prob = self.predict_regime_switch_probability()

        if switch_prob > self.config.switch_probability_threshold:
            return True, f"High regime switch probability: {switch_prob:.2%}"

        # Also check if current regime probability is low
        current_prob = self.regime_probabilities[self.current_regime.value]
        if current_prob < 0.5:
            return (
                True,
                f"Uncertain regime: {self.current_regime.name} probability only {current_prob:.2%}",
            )

        return False, ""

    def get_spread_adjustment(self) -> float:
        """
        Get spread adjustment multiplier for current regime.

        Returns:
            Multiplier to apply to base spread
        """
        return self.config.spread_multipliers.get(self.current_regime, 1.0)

    def get_regime_stats(self) -> Dict[str, Any]:
        """Get statistics about regime history."""
        if not self.regime_history:
            return {}

        regimes = [r[1] for r in self.regime_history]

        return {
            "total_observations": len(regimes),
            "regime_distribution": {
                regime.name: regimes.count(regime) / len(regimes) for regime in RegimeState
            },
            "current_regime": self.current_regime.name,
            "current_regime_probability": self.regime_probabilities[self.current_regime.value],
            "regime_switch_probability": self.predict_regime_switch_probability(),
            "model_fitted": self._is_fitted,
            "samples_since_last_training": self._sample_count - self._last_training_sample,
            "current_regime_run_length": self._current_regime_run_length,
            "training_failures": self._training_failures,
            "last_training_error": self._last_training_error,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._returns_buffer.clear()
        self._features_buffer.clear()
        self._timestamps.clear()
        self._sample_count = 0
        self._last_training_sample = 0
        self._is_fitted = False
        self._feature_mean = None
        self._feature_scale = None
        self._training_failures = 0
        self._last_training_error = ""
        self.current_regime = RegimeState.MEDIUM
        self.regime_probabilities = np.array([1 / 3, 1 / 3, 1 / 3])
        self._current_regime_run_length = 0
        self.regime_history.clear()
        self._init_model()


class SimpleThresholdRegimeDetector:
    """
    Simple threshold-based regime detector (fallback for HMM).

    Uses realized volatility thresholds to classify regimes.
    """

    def __init__(
        self, thresholds: Optional[List[float]] = None, annualization_periods: float = 365 * 24 * 60
    ):
        self.thresholds = thresholds or [0.3, 0.6]  # 30%, 60% annualized vol
        self.annualization_periods = annualization_periods
        self._returns_buffer: Deque[float] = deque(maxlen=100)
        self.current_regime: RegimeState = RegimeState.MEDIUM

    def update(self, returns: float) -> RegimeState:
        """Update with new return and classify regime."""
        self._returns_buffer.append(returns)

        if len(self._returns_buffer) < 10:
            return RegimeState.MEDIUM

        # Calculate realized volatility
        vol = float(np.std(self._returns_buffer) * np.sqrt(self.annualization_periods))

        # Classify
        if vol < self.thresholds[0]:
            self.current_regime = RegimeState.LOW
        elif vol < self.thresholds[1]:
            self.current_regime = RegimeState.MEDIUM
        else:
            self.current_regime = RegimeState.HIGH

        return self.current_regime

    def get_spread_adjustment(self) -> float:
        """Get spread multiplier."""
        multipliers = {RegimeState.LOW: 0.8, RegimeState.MEDIUM: 1.0, RegimeState.HIGH: 1.5}
        return multipliers.get(self.current_regime, 1.0)
