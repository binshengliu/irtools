#!/usr/bin/env python3
import sys
import numpy as np

from irtools.rank_metrics import ndcg_at_k, average_precision


class TrecQrels:
    def __init__(self, qrels, depth=None, method='reciprocal_rank'):
        if isinstance(qrels, str):
            self.qrels = {}
            with open(qrels, 'r') as f:
                for line in f:
                    qno, _, dno, rel = line.split()
                    self.qrels.setdefault(qno, {}).setdefault(dno, {})
                    self.qrels[qno][dno] = int(rel)
        elif isinstance(qrels, dict):
            self.qrels = qrels
        else:
            assert False
        self.depth = depth
        self.method = method

    def __call__(self, qno, sorted_dnos):
        func = getattr(self, self._normalize(self.method))
        return func(qno, sorted_dnos)

    def _normalize(self, method):
        if method == 'recip_rank':
            return 'reciprocal_rank'
        return method

    def set_method(self, method):
        self.method = str(method)

    def set_depth(self, depth):
        self.depth = int(depth)

    def relevance(self, qno, pid):
        return self.qrels.get(str(qno), {}).get(str(pid), 0)

    def describe(self, file=sys.stderr):
        has_rel = len([
            k for k, v in self.qrels.items()
            if np.count_nonzero(list(v.values())) >= 1
        ])
        print(
            '{} topics in qrels. {} have rel docs.'.format(
                len(self.qrels), has_rel),
            file=file)

    def __contains__(self, el):
        return el in self.qrels

    def select_by_min_rels(self, min_nrels):
        assert False, 'needs update'
        qrels = {
            k: v
            for k, v in self.qrels.items()
            if np.count_nonzero(list(v.values())) >= min_nrels
        }
        return TrecQrels(qrels)

    def select_by_rels(self, nrels):
        assert False, 'needs update'
        qrels = {
            k: v
            for k, v in self.qrels.items()
            if np.count_nonzero(list(v.values())) == nrels
        }
        return TrecQrels(qrels)

    def select_by_max_rels(self, max_nrels):
        assert False, 'needs update'
        qrels = {
            k: v
            for k, v in self.qrels.items()
            if np.count_nonzero(list(v.values())) <= max_nrels
        }
        return TrecQrels(qrels)

    def select_by_qnos(self, qnos):
        if isinstance(qnos, str):
            qnos = [qnos]
        qrels = {k: v for k, v in self.qrels.items() if k in qnos}
        return TrecQrels(qrels, self.depth, self.method)

    def select_by_qno(self, qno):
        qrels = {qno: self.qrels[qno]}
        return TrecQrels(qrels, self.depth, self.method)

    def reciprocal_rank(self, qno, sorted_dnos):
        # assert qno in self.qrels, 'No judgments for {}'.format(qno)
        if qno not in self.qrels:
            return None
        rels = [
            1 if self.qrels[qno].get(dno, 0) else 0
            for dno in sorted_dnos[:self.depth]
        ]
        try:
            score = 1 / (rels.index(1) + 1)
        except ValueError:
            score = 0

        return score

    def ndcg(self, qno, sorted_dnos):
        # assert qno in self.qrels, 'No judgments for {}'.format(qno)
        if qno not in self.qrels:
            return None
        rels = [
            self.qrels[qno].get(dno, 0) for dno in sorted_dnos[:self.depth]
        ]

        score = ndcg_at_k(rels, self.depth)

        return score

    def map(self, qno, sorted_dnos):
        # assert qno in self.qrels, 'No judgments for {}'.format(qno)
        if qno not in self.qrels:
            return None
        rels = [
            self.qrels[qno].get(dno, 0) for dno in sorted_dnos[:self.depth]
        ]

        score = average_precision(rels)

        return score

    def get(self, qno, dno, default=0):
        return self.qrels.get(qno, {}).get(dno, default)
