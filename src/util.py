from typing import Any


def idx_max(xs: list[Any]) -> int:
    max = 0
    idx = -1
    for i, x in enumerate(xs):
        if x > max:
            max = x
            idx = i
    return idx
