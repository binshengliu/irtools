from __future__ import annotations

import os
from collections import OrderedDict
from typing import AnyStr, Iterator, Tuple

from more_itertools import with_iter


class TrecQuery:
    def __init__(self, path: os.PathLike[AnyStr]):
        self._qno_map = OrderedDict(
            line.strip("\n").split("\t") for line in with_iter(open(path, "r"))
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._qno_map.items())

    def __getitem__(self, qno: str) -> str:
        query: str = self._qno_map[qno]
        return query

    def __len__(self) -> int:
        return len(self._qno_map)
