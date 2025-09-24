from edgebettor_ml.odds.pricing import expected_value_per_dollar


def test_ev_sign():
    # With p higher than implied (here use payoff implicitly), EV should be positive
    ev = expected_value_per_dollar(0.6, -110)
    assert ev > 0
    # With p lower than implied, EV should be negative
    ev2 = expected_value_per_dollar(0.4, -110)
    assert ev2 < 0


