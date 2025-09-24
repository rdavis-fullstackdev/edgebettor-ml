from __future__ import annotations

from typing import Optional


def american_to_implied_probability(odds: int | float | None) -> Optional[float]:
    """Convert American odds to implied win probability (no vig removed).

    Returns None if odds is None or 0.
    """
    if odds is None:
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / ((-odds) + 100.0)


def payoff_per_dollar(odds: int | float) -> float:
    """Return gross payoff (profit) per $1 stake for American odds.

    Examples:
      +150 -> 1.50
      -120 -> 0.8333...
    """
    if odds > 0:
        return odds / 100.0
    else:
        return 100.0 / (-odds)


def expected_value_per_dollar(win_probability: float, odds: int | float) -> float:
    """Compute EV per $1 stake.

    EV = p * payoff - (1 - p) * 1
    where payoff is the profit per $1 stake if the bet wins.
    """
    p = float(win_probability)
    payoff = payoff_per_dollar(odds)
    return p * payoff - (1.0 - p) * 1.0


def edge(win_probability: float, implied_probability: float) -> float:
    """Edge = model probability - implied probability."""
    return float(win_probability) - float(implied_probability)


def price_option(win_probability: float, odds: int | float) -> dict:
    """Convenience helper to compute pricing metrics for a single option.

    Returns dict with keys: implied, payoff, ev, edge
    """
    implied = american_to_implied_probability(odds)
    payoff = payoff_per_dollar(odds)
    ev = expected_value_per_dollar(win_probability, odds)
    edg = edge(win_probability, implied) if implied is not None else None
    return {"implied": implied, "payoff": payoff, "ev": ev, "edge": edg}


