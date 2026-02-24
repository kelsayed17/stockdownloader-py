"""Hidden Markov Model regime detector for market state classification.

Uses Gaussian HMM to classify market regimes (e.g. bull, bear, sideways)
based on return and volatility features.  Trained regime probabilities
are used as ML features alongside technical indicators.

Requires ``hmmlearn`` (pip install hmmlearn).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Regime classification for a single bar."""

    regime_id: int           # 0, 1, 2
    prob_regime_0: float     # probability of regime 0 (typically bull)
    prob_regime_1: float     # probability of regime 1 (typically bear)
    prob_regime_2: float     # probability of regime 2 (typically sideways)
    regime_duration: int     # consecutive bars in current regime


class HMMRegimeDetector:
    """Fits a Gaussian HMM to price data and classifies market regimes.

    Parameters
    ----------
    n_regimes:
        Number of hidden states (default: 3 for bull/bear/sideways).
    lookback:
        Number of bars used for rolling volatility feature.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 20,
        random_state: int = 42,
    ) -> None:
        self._n_regimes = n_regimes
        self._lookback = lookback
        self._random_state = random_state
        self._model = None
        self._regime_labels: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, closes: list[float]) -> list[RegimeSnapshot]:
        """Fit HMM on full close series and return per-bar regime snapshots.

        Parameters
        ----------
        closes:
            List of close prices (chronological).

        Returns
        -------
        list[RegimeSnapshot]
            One snapshot per bar.  The first ``lookback`` bars will have
            regime_id=0 and uniform probabilities (insufficient history).
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning(
                "hmmlearn not installed — returning default regimes"
            )
            return self._default_regimes(len(closes))

        n = len(closes)
        if n < self._lookback + 10:
            logger.warning(
                "Not enough data for HMM (%d bars, need %d)",
                n, self._lookback + 10,
            )
            return self._default_regimes(n)

        # Build observation matrix: [log_return, rolling_vol, vol_of_vol]
        arr = np.array(closes, dtype=np.float64)
        log_returns = np.diff(np.log(arr))  # (n-1,)

        # Pad first element
        log_returns = np.concatenate([[0.0], log_returns])

        # Rolling volatility (std of returns over lookback window)
        rolling_vol = np.zeros(n)
        for i in range(self._lookback, n):
            rolling_vol[i] = np.std(log_returns[i - self._lookback: i])

        # Forward-fill early bars
        if self._lookback < n:
            rolling_vol[:self._lookback] = rolling_vol[self._lookback]

        # Observations: 2-D feature matrix
        X = np.column_stack([log_returns, rolling_vol])

        # Fit HMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=self._n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=self._random_state,
                init_params="stmc",
            )
            try:
                model.fit(X)
            except Exception as exc:
                logger.warning("HMM fitting failed: %s", exc)
                return self._default_regimes(n)

        self._model = model

        # Decode: predict most likely state sequence
        try:
            _, state_sequence = model.decode(X)
        except Exception as exc:
            logger.warning("HMM decode failed: %s", exc)
            return self._default_regimes(n)

        # Get state probabilities per bar
        try:
            posteriors = model.predict_proba(X)
        except Exception:
            posteriors = np.full((n, self._n_regimes), 1.0 / self._n_regimes)

        # Label regimes by mean return (highest mean return = bull)
        self._label_regimes(model, X, state_sequence)

        # Build snapshots
        snapshots: list[RegimeSnapshot] = []
        duration = 0
        prev_state = -1

        for i in range(n):
            state = int(state_sequence[i])
            if state == prev_state:
                duration += 1
            else:
                duration = 1
                prev_state = state

            probs = posteriors[i] if i < len(posteriors) else np.full(
                self._n_regimes, 1.0 / self._n_regimes,
            )

            # Ensure we always have exactly 3 probabilities
            p0 = float(probs[0]) if len(probs) > 0 else 0.33
            p1 = float(probs[1]) if len(probs) > 1 else 0.33
            p2 = float(probs[2]) if len(probs) > 2 else 0.34

            snapshots.append(
                RegimeSnapshot(
                    regime_id=state,
                    prob_regime_0=p0,
                    prob_regime_1=p1,
                    prob_regime_2=p2,
                    regime_duration=duration,
                )
            )

        logger.info(
            "HMM fitted: %d regimes, labels=%s",
            self._n_regimes, self._regime_labels,
        )
        return snapshots

    @property
    def regime_labels(self) -> dict[int, str]:
        """Mapping from regime ID to human-readable label."""
        return dict(self._regime_labels)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _label_regimes(
        self, model: object, X: np.ndarray, states: np.ndarray,
    ) -> None:
        """Assign bull/bear/sideways labels based on mean return per state."""
        mean_returns: dict[int, float] = {}
        for s in range(self._n_regimes):
            mask = states == s
            if mask.sum() > 0:
                mean_returns[s] = float(X[mask, 0].mean())
            else:
                mean_returns[s] = 0.0

        sorted_states = sorted(mean_returns, key=mean_returns.get)  # type: ignore[arg-type]

        labels = ["bear", "sideways", "bull"]
        if self._n_regimes == 2:
            labels = ["bear", "bull"]

        self._regime_labels = {}
        for i, state_id in enumerate(sorted_states):
            if i < len(labels):
                self._regime_labels[state_id] = labels[i]
            else:
                self._regime_labels[state_id] = f"regime_{state_id}"

    def _default_regimes(self, n: int) -> list[RegimeSnapshot]:
        """Return neutral default regimes when HMM cannot be fitted."""
        p = 1.0 / max(self._n_regimes, 1)
        return [
            RegimeSnapshot(
                regime_id=0,
                prob_regime_0=p,
                prob_regime_1=p,
                prob_regime_2=1.0 - 2 * p if self._n_regimes >= 3 else 0.0,
                regime_duration=i + 1,
            )
            for i in range(n)
        ]


# ------------------------------------------------------------------
# Feature names for integration with FeatureExtractor
# ------------------------------------------------------------------

HMM_FEATURE_NAMES: list[str] = [
    "hmm_regime_id",
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_prob_sideways",
    "hmm_regime_duration_norm",
]
"""5 HMM regime features added to the ML feature vector."""
