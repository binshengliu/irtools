from __future__ import annotations

import os
from typing import AnyStr, List, Tuple

from more_itertools import with_iter


class QrelLine:
    def __init__(self, line: str):
        self.qno, self.dno, self.rel = self.parse_line(line)

    @staticmethod
    def parse_line(line: str) -> Tuple[str, str, int]:
        fields = line.split()
        if len(fields) == 4:
            qno, _, dno, rel = fields
            return qno, dno, int(rel)
        else:
            raise ValueError("Unknown run format")

    def __str__(self) -> str:
        return f"{self.qno}\t0\t{self.dno}\t{self.rel}\n"

    def __repr__(self) -> str:
        return str(self)


class TrecQrel:
    def __init__(self, path: os.PathLike[AnyStr], default_rel: int = 0):
        lines = [QrelLine(line) for line in with_iter(open(path, "r"))]
        self._qno_map = {x.qno: {x.dno: x.rel} for x in lines}
        self._dno_map = {x.dno: {x.qno: x.rel} for x in lines}
        self._default_rel = default_rel

    def lookup(self, qno: str, dno: str) -> int:
        return self._qno_map.get(qno, {}).get(dno, self._default_rel)

    def relevant_docs(self, qno: str) -> List[str]:
        return list(k for k, v in self._qno_map.get(qno, {}).items() if v > 0)

    def relevant_qnos(self, dno: str) -> List[str]:
        return list(k for k, v in self._dno_map.get(dno, {}).items() if v > 0)
