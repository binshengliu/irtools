from __future__ import annotations

import os
from collections import OrderedDict
from typing import AnyStr, Iterator, Tuple

import numpy as np
from more_itertools import with_iter


class TrecEval:
    def __init__(self, path: os.PathLike[AnyStr], default: float = 0.0):
        lines = [line.strip("\n").split() for line in with_iter(open(path, "r"))]
        assert all(len(x) == 3 for x in lines), "Malformed eval file"
        self._metric = lines[0][0]
        self._qno_map = OrderedDict((x[1], float(x[2])) for x in lines)
        self._default = default
        agg = round(np.mean(list(self._qno_map.values())), 4)
        if "all" in self._qno_map:
            self._agg = self._qno_map.pop("all")
            assert np.isclose(self._agg, agg), "The aggregated value is incorrect"
        else:
            self._agg = agg

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        return iter(self._qno_map.items())

    def __getitem__(self, qno: str) -> float:
        return self._qno_map.get(qno, self._default)

    def get(self, qno: str) -> float:
        return self._qno_map.get(qno, self._default)

    def __len__(self) -> int:
        return len(self._qno_map)
