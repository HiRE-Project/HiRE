"""Potential differences; return construction remains the RL backend's responsibility."""


def potential_difference(current, next, discount, weight=1.0, normalizer=1.0):
    """Works with tensors or arrays; discounts use the transition's temporal unit."""
    return (discount * next - current) * weight / normalizer
