"""GMESignalFusion -- 4-pillar scoring and regime detection.

Combines options analytics from the StateEngine with alt data and ML
scores into a composite score plus regime classification.

The four pillars are:
1. **options_flow** -- IV percentile, skew, GEX proximity, OI change
2. **volume_premium** -- P/C volume ratio, premium imbalance
3. **cycle_timing** -- T+35 countdown, SI change, dark pool change
4. **momentum** -- ML model score centred at 0 and scaled to +/-2

Each pillar produces a z-score-like value.  A weighted composite is
computed, and anomaly flags are raised for any pillar exceeding
+/- ``ANOMALY_THRESHOLD`` standard deviations.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from stockdownloader.gme.options.scorecard import GMEDailyScorecard

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "options_flow": 0.30,
    "volume_premium": 0.20,
    "cycle_timing": 0.30,
    "momentum": 0.20,
}

ANOMALY_THRESHOLD: float = 2.0


def _clamp(value: float, lo: float = -5.0, hi: float = 5.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


class GMESignalFusion:
    """Fuse multiple signal pillars into a daily composite scorecard.

    Parameters
    ----------
    weights:
        Optional mapping of pillar name to weight.  If *None*,
        :data:`DEFAULT_WEIGHTS` is used.  Weights are normalised
        internally so they sum to 1.0.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = dict(weights or DEFAULT_WEIGHTS)
        total = sum(w.values()) or 1.0
        self._weights = {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------
    # Pillar scoring
    # ------------------------------------------------------------------

    def _score_options(self, options: dict[str, Any]) -> float:
        """Score the options-flow pillar.

        Components (each mapped to roughly +/-2 range then averaged):

        * **IV percentile** -- 0.5 is neutral; 0 or 1 are extremes.
          Mapped: ``(iv_percentile - 0.5) * 4`` so 0->-2, 1->+2.
        * **IV skew (25d, 30-day)** -- positive skew = puts expensive
          (bearish hedging).  Scaled by 10 to bring typical 0.05-0.20
          range into +/-2.
        * **GEX proximity** -- distance from spot (flip price) to call
          wall, normalised.  A tight spread is bullish.
        * **OI change** -- put/call OI ratio change.  Rising P/C ratio
          is bearish, scaled by 20.
        """
        iv_pct = options.get("iv_percentile", 0.5)
        iv_skew = options.get("iv_skew_25d_30", 0.0)
        gex_flip = options.get("gex_flip_price", 0.0)
        call_wall = options.get("call_wall", 0.0)
        pc_oi_change = options.get("pc_oi_ratio_change", 0.0)

        # IV percentile: map [0,1] to [-2,+2]
        iv_score = (iv_pct - 0.5) * 4.0

        # IV skew: positive skew (puts expensive) is bearish, invert
        skew_score = _clamp(-iv_skew * 10.0, -2.0, 2.0)

        # GEX proximity: how far flip price is below call wall
        if call_wall > 0 and gex_flip > 0:
            proximity = (call_wall - gex_flip) / call_wall
            gex_score = _clamp((0.2 - proximity) * 10.0, -2.0, 2.0)
        else:
            gex_score = 0.0

        # OI change: rising P/C ratio is bearish
        oi_score = _clamp(-pc_oi_change * 20.0, -2.0, 2.0)

        return _clamp((iv_score + skew_score + gex_score + oi_score) / 4.0)

    def _score_volume(self, options: dict[str, Any]) -> float:
        """Score the volume-premium pillar.

        * **P/C volume ratio** -- >1 means more puts traded (bearish).
          Mapped: ``(1.0 - pc_volume_ratio) * 2`` so 0.5->+1, 1.5->-1.
        * **Premium imbalance** -- positive = more call premium (bullish).
          Already in [-1, +1]; scale by 2.
        """
        pc_vol = options.get("pc_volume_ratio", 1.0)
        prem_imb = options.get("premium_imbalance", 0.0)

        vol_score = _clamp((1.0 - pc_vol) * 2.0, -2.0, 2.0)
        prem_score = _clamp(prem_imb * 2.0, -2.0, 2.0)

        return _clamp((vol_score + prem_score) / 2.0)

    def _score_cycles(self, alt: dict[str, Any]) -> float:
        """Score the cycle-timing pillar.

        * **T+35 countdown** -- days until FTD settlement.  Lower is
          hotter.  Mapped: ``(15 - countdown) / 7.5`` so 0->+2, 15->0,
          30->-2.
        * **SI change (2wk)** -- positive = short interest rising.
          Scaled by 0.5 to map a 4pp change to +2.
        * **Dark pool ratio change** -- positive means increasing dark
          pool activity (bearish signal).  Scaled by 10.
        """
        countdown = alt.get("ftd_t35_countdown", 35)
        si_change = alt.get("si_change_2wk", 0.0)
        dp_change = alt.get("dark_pool_ratio_change", 0.0)

        ftd_score = _clamp((15.0 - countdown) / 7.5, -2.0, 2.0)
        si_score = _clamp(si_change * 0.5, -2.0, 2.0)
        dp_score = _clamp(-dp_change * 10.0, -2.0, 2.0)

        return _clamp((ftd_score + si_score + dp_score) / 3.0)

    def _score_momentum(self, ml_score: float) -> float:
        """Score the momentum pillar.

        The ML model emits a probability in [0, 1].  Centre at 0 and
        scale to +/-2:  ``(ml_score - 0.5) * 4``.
        """
        return _clamp((ml_score - 0.5) * 4.0, -2.0, 2.0)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(
        self,
        options: dict[str, Any],
        alt: dict[str, Any],
        composite: float,
    ) -> str:
        """Classify the current market regime.

        Returns one of ``"squeeze"``, ``"gamma_ramp"``, ``"cycle_hot"``,
        or ``"neutral"``.

        Regime rules (evaluated in priority order):

        * **squeeze** -- ``iv_percentile < 0.30`` AND
          ``gex_concentration > 0.6``.
        * **gamma_ramp** -- ``gex_concentration > 0.5`` AND
          ``abs(composite) > 1.0``.
        * **cycle_hot** -- ``ftd_t35_countdown < 5`` AND
          (``abs(si_change_2wk) > 2.0`` OR
          ``abs(dark_pool_ratio_change) > 0.1``).
        * **neutral** -- none of the above.
        """
        iv_pct = options.get("iv_percentile", 0.5)
        gex_conc = options.get("gex_concentration", 0.0)

        ftd_countdown = alt.get("ftd_t35_countdown", 35)
        si_change = alt.get("si_change_2wk", 0.0)
        dp_change = alt.get("dark_pool_ratio_change", 0.0)

        # Priority 1: squeeze
        if iv_pct < 0.30 and gex_conc > 0.6:
            return "squeeze"

        # Priority 2: gamma ramp
        if gex_conc > 0.5 and abs(composite) > 1.0:
            return "gamma_ramp"

        # Priority 3: cycle hot
        if ftd_countdown < 5 and (abs(si_change) > 2.0 or abs(dp_change) > 0.1):
            return "cycle_hot"

        return "neutral"

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _flag_anomalies(self, pillars: dict[str, float]) -> list[str]:
        """Flag any pillar whose absolute value exceeds the threshold.

        Parameters
        ----------
        pillars:
            Mapping of pillar name to z-score-like value.

        Returns
        -------
        List of human-readable anomaly descriptions.
        """
        flags: list[str] = []
        for name, score in pillars.items():
            if abs(score) > ANOMALY_THRESHOLD:
                direction = "high" if score > 0 else "low"
                flags.append(
                    f"{name} anomaly: score {score:+.2f} "
                    f"({direction}, threshold +/-{ANOMALY_THRESHOLD})"
                )
        return flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_daily_scorecard(
        self,
        trade_date: date,
        options_state: dict[str, Any],
        alt_data: dict[str, Any],
        ml_score: float = 0.5,
    ) -> GMEDailyScorecard:
        """Produce a :class:`GMEDailyScorecard` for a single trading day.

        Parameters
        ----------
        trade_date:
            The date being scored.
        options_state:
            Dictionary of options metrics (from
            :class:`OptionsStateEngine`).
        alt_data:
            Dictionary of alternative data (FTD, SI, dark pool, etc.).
        ml_score:
            ML model probability in [0, 1].  Default 0.5 (neutral).

        Returns
        -------
        A frozen :class:`GMEDailyScorecard`.
        """
        pillar_scores: dict[str, float] = {
            "options_flow": self._score_options(options_state),
            "volume_premium": self._score_volume(options_state),
            "cycle_timing": self._score_cycles(alt_data),
            "momentum": self._score_momentum(ml_score),
        }

        # Weighted composite
        composite = sum(
            self._weights.get(name, 0.0) * score
            for name, score in pillar_scores.items()
        )

        # Regime classification -- merge options_state keys needed for
        # regime detection with alt_data so _detect_regime has full access.
        regime_options = {
            "iv_percentile": options_state.get("iv_percentile", 0.5),
            "gex_concentration": options_state.get("gex_concentration", 0.0),
        }
        regime = self._detect_regime(regime_options, alt_data, composite)

        # Anomaly flags
        anomaly_flags = self._flag_anomalies(pillar_scores)

        return GMEDailyScorecard(
            date=trade_date,
            pillar_scores=pillar_scores,
            composite=composite,
            regime=regime,
            anomaly_flags=anomaly_flags,
        )
