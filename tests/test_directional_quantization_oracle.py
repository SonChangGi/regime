"""Compare joint serialization with an exhaustive, exact integer-grid oracle."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from regime_lab.analysis.directional_coherence import (
    _polygon_projection,
    _quantize_joint_pair,
)
from regime_lab.payload import PROBABILITY_SCALE


@pytest.mark.parametrize("anchor", [0, 1, 35_647_173, PROBABILITY_SCALE - 6])
@pytest.mark.parametrize("delta4,delta13", list(product(range(4), repeat=2)))
def test_quantized_projection_minimizes_original_objective_over_entire_feasible_grid(
    anchor, delta4, delta13,
):
    # Exhaust the full feasible lattice, including line and point degeneracies.
    feasible = [
        (x, y)
        for x in range(anchor, anchor + delta4 + 1)
        for y in range(x, x + delta13 + 1)
    ]
    # Half-unit targets exercise ties; exterior points exercise every boundary.
    offsets = (-2, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 8)
    for dx, dy in product(offsets, repeat=2):
        target = np.array([anchor + dx, anchor + dy]) / PROBABILITY_SCALE
        projected = _polygon_projection(
            target, anchor / PROBABILITY_SCALE,
            delta4 / PROBABILITY_SCALE, delta13 / PROBABILITY_SCALE,
        )
        actual = _quantize_joint_pair(projected, target, anchor, delta4, delta13)
        # Exact arithmetic avoids losing one-unit objective differences at large
        # anchors. Use the actual represented target, without a second projection.
        tx, ty = map(Fraction.from_float, target * PROBABILITY_SCALE)

        def original_objective(pair):
            return (pair[0] - tx) ** 2 + (pair[1] - ty) ** 2

        assert actual in feasible
        assert original_objective(actual) == min(map(original_objective, feasible)), (
            anchor, delta4, delta13, dx, dy, actual,
        )
        assert _quantize_joint_pair(projected, target, anchor, delta4, delta13) == actual
