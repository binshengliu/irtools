from __future__ import annotations

import os
from itertools import chain
from typing import AnyStr, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

from tqdm import tqdm


class RunLine:
    def __init__(self, line: str):
        (
            self.type,
            self.qno,
            self.vno,
            self.docid,
            self.rank,
            self.score,
            self.identity,
        ) = self.parse_line(line)

    @staticmethod
    def parse_line(line: str) -> Tuple[str, str, str, str, int, float, str]:
        fields = line.split()
        if len(fields) == 6:
            type_ = "indri"
            vno, _, docid, rank_str, score_str, identity = fields
            rank = int(rank_str)
            score = float(score_str)
        elif len(fields) == 3:
            type_ = "anserini"
            identity = ""
            vno, docid, rank_str = fields
            rank = int(rank_str)
            score = 1.0 / rank
        else:
            raise ValueError("Unknown run format")
        qno = vno.split("-")[0]
        return type_, qno, vno, docid, rank, score, identity

    def __str__(self) -> str:
        if self.type == "indri":
            return f"{self.vno} Q0 {self.docid} {self.rank} {self.score} indri\n"
        elif self.type == "anserini":
            return f"{self.vno}\t{self.docid}\t{self.rank}\n"
        else:
            assert False, f"Unknown type {self.type}"

    def __repr__(self) -> str:
        return str(self)


class TrecRunVno:
    def __init__(
        self, qno: str, vno: str, records: Sequence[RunLine] = [],
    ):
        "docstring"
        self.qno = qno
        self.vno = vno
        self.records = sorted(records, key=lambda x: x.score, reverse=True)
        for rank, record in enumerate(self.records, start=1):
            record.rank = rank

    @staticmethod
    def from_buffer(buffer: Sequence[str]) -> TrecRunVno:
        records = [RunLine(x) for x in buffer]
        return TrecRunVno.from_records(records)

    @staticmethod
    def from_records(records: Sequence[RunLine]) -> TrecRunVno:
        """qno, vno, docid, rank, score"""
        qno, vno = records[0].qno, records[0].vno
        assert all(qno == x.qno for x in records[1:])
        assert all(vno == x.vno for x in records[1:])
        return TrecRunVno(qno, vno, records)

    def __str__(self) -> str:
        buffer = "".join(map(str, self.records))
        return buffer

    def first(self, num: int) -> TrecRunVno:
        return TrecRunVno(self.qno, self.vno, self.records[:num])

    def select_by_ranks(
        self, ranks: Iterable[int], zero_based: bool = False
    ) -> TrecRunVno:
        ranks = [x - 1 for x in ranks] if not zero_based else ranks
        records = [self.records[x] for x in ranks]
        return TrecRunVno(self.qno, self.vno, records)

    def __iter__(self) -> Iterator[RunLine]:
        return self.iter_lines()

    def iter_lines(self) -> Iterator[RunLine]:
        return iter(self.records)

    def __repr__(self) -> str:
        return str(self)


class TrecRunQno:
    def __init__(self, qno: str, vno_map: Mapping[str, TrecRunVno] = {}):
        "docstring"
        self.qno = qno
        self.vno_map = vno_map

    @staticmethod
    def from_records(records: Sequence[RunLine]) -> TrecRunQno:
        vno_records: Dict[str, List[RunLine]] = {}
        for record in records:
            qno = record.qno
            vno_records.setdefault(record.vno, []).append(record)
        assert vno_records
        vno_map = {k: TrecRunVno.from_records(v) for k, v in vno_records.items()}
        return TrecRunQno(qno, vno_map)

    @staticmethod
    def from_buffer(buffer: Sequence[str]) -> TrecRunQno:
        records = [RunLine(x) for x in buffer]
        return TrecRunQno.from_records(records)

    def original(self) -> TrecRunQno:
        if self.qno in self.vno_map:
            return TrecRunQno(self.qno, {self.qno: self.vno_map[self.qno]})
        else:
            return TrecRunQno(self.qno, {})

    def variants(self) -> TrecRunQno:
        return TrecRunQno(
            self.qno, {k: v for k, v in self.vno_map.items() if k != self.qno}
        )

    def vnos(self) -> Set[str]:
        return set(self.vno_map.keys())

    def select_by_vnos(self, vnos: Iterable[str]) -> TrecRunQno:
        vno_map = {k: v for k, v in self.vno_map.items() if k in vnos}
        return TrecRunQno(self.qno, vno_map)

    def first(self, num: int) -> TrecRunQno:
        return TrecRunQno(self.qno, {k: v.first(num) for k, v in self.vno_map.items()})

    def select_by_ranks(
        self, ranks: Iterable[int], zero_based: bool = False
    ) -> TrecRunQno:
        return TrecRunQno(
            self.qno,
            {k: v.select_by_ranks(ranks, zero_based) for k, v in self.vno_map.items()},
        )

    def __getitem__(self, key: str) -> TrecRunVno:
        return self.vno_map[key]

    def __str__(self) -> str:
        return "".join(map(str, self.vno_map.values()))

    def __iter__(self) -> Iterator[TrecRunVno]:
        return iter(self.vno_map.values())

    def iter_lines(self) -> Iterator[RunLine]:
        return chain.from_iterable([x.iter_lines() for x in self.vno_map.values()])

    def __repr__(self) -> str:
        return str(self)


class TrecRun:
    def __init__(self, qno_map: Mapping[str, TrecRunQno]):
        self.qno_map = qno_map

    @staticmethod
    def from_buffer(buffer: Sequence[str], progress: bool = False) -> TrecRun:
        qno_records: Dict[str, List[RunLine]] = {}
        if progress:
            buffer = tqdm(buffer, desc="Parse")
        for line in buffer:
            record = RunLine(line)
            qno_records.setdefault(record.qno, []).append(record)
        qno_map = {k: TrecRunQno.from_records(v) for k, v in qno_records.items()}
        return TrecRun(qno_map)

    @staticmethod
    def from_file(path: os.PathLike[AnyStr], progress: bool = False) -> TrecRun:
        with open(path, "r") as f:
            buffer = f.readlines()
        return TrecRun.from_buffer(buffer, progress)

    def num_qnos(self) -> int:
        return len(self.qno_map)

    def first(self, num: int) -> TrecRun:
        return TrecRun({k: v.first(num) for k, v in self.qno_map.items()})

    def select_original(self) -> TrecRun:
        qno_map = {k: v.original() for k, v in self.qno_map.items()}
        return TrecRun(qno_map)

    def select_variants(self) -> TrecRun:
        qno_map = {k: v.variants() for k, v in self.qno_map.items()}
        return TrecRun(qno_map)

    def select_by_vnos(self, vnos: Iterable[str]) -> TrecRun:
        qno_map = {k: v.select_by_vnos(vnos) for k, v in self.qno_map.items()}
        return TrecRun(qno_map)

    def select_by_ranks(
        self, ranks: Iterable[int], zero_based: bool = False
    ) -> TrecRun:
        qno_map = {
            k: v.select_by_ranks(ranks, zero_based) for k, v in self.qno_map.items()
        }
        return TrecRun(qno_map)

    def __str__(self) -> str:
        return "".join(map(str, self.qno_map.values()))

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> Iterator[TrecRunQno]:
        return iter(self.qno_map.values())

    def iter_lines(self) -> Iterator[RunLine]:
        return chain.from_iterable([x.iter_lines() for x in self.qno_map.values()])
