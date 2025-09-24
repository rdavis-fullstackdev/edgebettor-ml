from edgebettor_ml.odds.pricing import (
    american_to_implied_probability,
    payoff_per_dollar,
    expected_value_per_dollar,
    edge,
)


def test_american_to_implied_probability_positive():
    # +150 -> 100/(150+100) = 0.4
    assert abs(american_to_implied_probability(150) - 0.4) < 1e-9


def test_american_to_implied_probability_negative():
    # -120 -> 120/(120+100) = 0.545454...
    p = american_to_implied_probability(-120)
    assert p is not None
    assert abs(p - (120 / 220)) < 1e-9


def test_payoff_per_dollar():
    assert abs(payoff_per_dollar(150) - 1.5) < 1e-9
    assert abs(payoff_per_dollar(-120) - (100 / 120)) < 1e-9


def test_expected_value_per_dollar():
    # p = 0.5, odds +100 => payoff 1.0 => EV = 0.5*1 - 0.5*1 = 0
    assert abs(expected_value_per_dollar(0.5, 100) - 0.0) < 1e-9
    # p = 0.6, odds -110 => payoff 100/110, EV positive but small
    ev = expected_value_per_dollar(0.6, -110)
    assert ev > 0


def test_edge():
    assert abs(edge(0.55, 0.50) - 0.05) < 1e-12


