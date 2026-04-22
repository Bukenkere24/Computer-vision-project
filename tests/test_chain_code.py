import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.chain_code import _decompose_step, contour_to_chain_code


def test_decompose():
    assert _decompose_step(2, 0) == [(1, 0), (1, 0)]
    assert _decompose_step(0, 0) == []


def test_chain_small_square():
    c = np.array(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], dtype=np.int32
    ).reshape(-1, 1, 2)
    ch = contour_to_chain_code(c)
    assert len(ch) > 0
    h = np.bincount(ch, minlength=8)
    assert h.sum() == len(ch)


if __name__ == "__main__":
    test_decompose()
    test_chain_small_square()
    print("tests ok")
