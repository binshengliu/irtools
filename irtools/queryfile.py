from __future__ import annotations

import os
from collections import OrderedDict
from typing import AnyStr, Iterator, Tuple

from more_itertools import with_iter


class TrecQuery:
    def __init__(self, qno_map: OrderedDict[str, str]):
        self._qno_map = qno_map

    @staticmethod
    def load(
        path: os.PathLike[AnyStr],
        sep: str = "\t",
        qid_field: int = 0,
        query_field: int = 1,
    ) -> TrecQuery:
        content = [line.strip("\n").split(sep) for line in with_iter(open(path, "r"))]
        qno_map = OrderedDict([(x[qid_field], x[query_field]) for x in content])
        return TrecQuery(qno_map)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._qno_map.items())

    def __getitem__(self, qno: str) -> str:
        query: str = self._qno_map[qno]
        return query

    def __len__(self) -> int:
        return len(self._qno_map)

    def head(self, n: int) -> TrecQuery:
        qno_map = OrderedDict((k, v) for k, v in list(self._qno_map.items())[:n])
        return TrecQuery(qno_map)

    def tail(self, n: int) -> TrecQuery:
        qno_map = OrderedDict((k, v) for k, v in list(self._qno_map.items())[n:])
        return TrecQuery(qno_map)

    def __repr__(self) -> str:
        return repr(self._qno_map)

    def __str__(self) -> str:
        return str(self._qno_map)
