from __future__ import annotations

import os
from collections import OrderedDict
from typing import AnyStr, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from more_itertools import first, with_iter


class TrecEval:
    def __init__(self, path: os.PathLike[AnyStr], default: float = 0.0):
        lines = [line.strip("\n").split() for line in with_iter(open(path, "r"))]
        assert all(len(x) == 3 for x in lines), "Malformed eval file"

        self._content: Dict[str, Dict[str, float]] = OrderedDict()
        self._agg: Dict[str, float] = OrderedDict()
        for metric, qid, value in lines:
            self._content.setdefault(metric, OrderedDict())
            if qid == "all":
                self._agg[metric] = float(value)
            else:
                self._content[metric][qid] = float(value)

        self._default = default

        # Check the qids are consistent across metrics
        first_qids = set(first(self._content.values()).keys())
        assert all(set(x.keys()) == first_qids for x in self._content.values())

    def __len__(self) -> int:
        return sum(len(x) for x in self._content.values())

    def __iter__(self) -> Iterator[Tuple[str, str, float]]:
        for metric, qid_values in self._content.items():
            for qid, value in qid_values.items():
                yield metric, qid, value

    def metrics(self) -> List[str]:
        return list(self._content.keys())

    def qids(self) -> List[str]:
        return list(first(self._content.values()).keys())

    def qid_value(self, qid: str, metric: Optional[str] = None) -> float:
        if metric is None:
            assert len(self._content) == 1, "metric can only be omitted when unique."
            return first(self._content.values()).get(qid, self._default)
        else:
            return self._content[metric].get(qid, self._default)

    def qid_values(self, qid: str) -> Dict[str, float]:
        return {metric: values[qid] for metric, values in self._content.items()}

    def agg_value(self, metric: Optional[str] = None) -> float:
        if metric is None:
            assert len(self._agg) == 1, "metric can only be omitted when unique"
            return first(self._agg.values())
        else:
            return self._agg[metric]

    def metric_values(self, metric: Optional[str] = None) -> List[float]:
        if metric is None:
            assert len(self._content) == 1, "metric can only be omitted when unique"
            return list(first(self._content.values()).values())
        else:
            return list(self._content[metric].values())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._content)
