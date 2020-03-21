from multiprocessing import Pool
from itertools import repeat, combinations, chain
from scipy.special import softmax
from more_itertools import unzip
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import random
import sys
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def tqdmf(path, *args, **kwargs):
    with tqdm(
            total=os.stat(path).st_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            *args,
            **kwargs) as bar:
        with open(path, 'r') as f:
            for line in f:
                bar.update(len(line))
                yield line


class TrecRunVno:
    def __init__(self, qno, vno, content=[], scores={}):
        "docstring"
        self.qno = qno
        self.vno = vno
        self.content = content
        self.scores = scores

    @staticmethod
    def from_buffer(buffer):
        scores, qno, vno = {}, None, None

        for line in buffer:
            tmpqno, tmpvno, docid, rank, score = parse_line(line)
            qno, vno = tmpqno, tmpvno
            scores[docid] = float(score)

        content = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return TrecRunVno(qno, vno, content, scores)

    @staticmethod
    def from_records(records):
        """qno, vno, docid, rank, score"""
        qno, vno = records[0][:2]
        scores = {x[2]: float(x[4]) for x in records}
        content = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return TrecRunVno(qno, vno, content, scores)

    @staticmethod
    def from_doc_scores(vno, doc_scores):
        qno, vno = vno.split('-')[0], vno
        scores = {}
        for doc, score in doc_scores:
            scores[doc] = float(score)

        content = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return TrecRunVno(qno, vno, content, scores)

    def is_variant(self):
        return self.qno != self.vno

    def eval(self, qrel):
        score = qrel(self.qno, self.sorted_dnos())
        return {self.vno: score}

    def sorted_dnos(self):
        return self.content

    def to_trec(self):
        buffer = ''
        for rank, dno in enumerate(self.content, 1):
            score = self.scores[dno]
            buffer += f'{self.vno} Q0 {dno} {rank} {score} indri\n'
        return buffer


def all_combinations(any_list):
    return chain.from_iterable(
        combinations(any_list, i + 1) for i in range(len(any_list)))


def reweight(weights, index, target_weight):
    weights *= (1 - target_weight)
    weights[index] = target_weight
    return weights


class TrecRunQno:
    def __init__(self, qno, vno_map={}):
        "docstring"
        self.qno = qno
        self.vno_map = vno_map

    @staticmethod
    def from_records(records):
        qno = records[0][0]
        vno_map = {}
        for record in records:
            vno_map.setdefault(record[1], []).append(record)
        vno_map = {k: TrecRunVno.from_records(v) for k, v in vno_map.items()}
        return TrecRunQno(qno, vno_map)

    @staticmethod
    def from_buffer(buffer):
        qno = None
        vno_map = {}
        for line in buffer:
            vno = line.split(maxsplit=1)[0]
            qno = vno.split('-')[0]
            vno_map.setdefault(vno, []).append(line)
        vno_map = {k: TrecRunVno.from_buffer(v) for k, v in vno_map.items()}
        return TrecRunQno(qno, vno_map)

    @staticmethod
    def from_doc_scores(vno, doc_scores):
        vno_map = {vno: TrecRunVno.from_doc_scores(vno, doc_scores)}
        qno = vno.split('-')[0]
        return TrecRunQno(qno, vno_map)

    def eval(self, qrel):
        scores = {}
        for runvno in self.vno_map.values():
            scores.update(runvno.eval(qrel))

        if not scores:
            scores = {self.qno: 0.0}
        return scores

    def original(self):
        orig_run = None
        for runvno in self.vno_map.values():
            if not runvno.is_variant():
                orig_run = TrecRunQno(self.qno, {self.qno: runvno})

        return orig_run

    def variants(self):
        vno_map = {}
        for runvno in self.vno_map.values():
            if runvno.is_variant():
                vno_map[runvno.vno] = runvno

        return TrecRunQno(self.qno, vno_map)

    def vnos(self):
        return set(self.vno_map.keys())

    def select_by_vnos(self, vnos, keep_origin_if_none=False):
        vno_map = {k: v for k, v in self.vno_map.items() if k in vnos}
        if not vno_map and keep_origin_if_none:
            vno_map = {self.qno: self.vno_map[self.qno]}
        return TrecRunQno(self.qno, vno_map)

    def individual(self, pos, scores):
        vnos = sorted(
            self.vno_map.keys(), key=lambda x: scores[x], reverse=True)
        pos = min(pos, len(vnos))
        vno = vnos[pos - 1]
        result = TrecRunQno(self.qno, {vno: self.vno_map[vno]})

        return result

    def fuse(self,
             cutoff=None,
             scores=None,
             fuse_func=None,
             weighted=False,
             always_include_orig=False,
             orig_weight=None):
        if len(self.vno_map) <= 1:
            return self

        vnos = list(self.vno_map.keys())
        if always_include_orig and self.qno in self.vno_map:
            vnos = list(set(vnos) - {self.qno})
            vnos = sorted(vnos, key=lambda x: scores[x], reverse=True)
            if cutoff is not None:
                vnos = vnos[:cutoff - 1] + [self.qno]
            vnos = sorted(vnos, key=lambda x: scores[x], reverse=True)
            weights = softmax([scores[x] for x in vnos]) if weighted else None
            if orig_weight is not None and 0 <= orig_weight <= 1:
                weights = reweight(weights, vnos.index(self.qno), orig_weight)
        else:
            weights = [1 for x in vnos]
            # vnos = sorted(vnos, key=lambda x: scores[x], reverse=True)[:cutoff]
            # weights = softmax([scores[x] for x in vnos]) if weighted else None

        fused = fuse_func([self.vno_map[x].sorted_dnos() for x in vnos],
                          weights)

        result = TrecRunQno.from_doc_scores(self.qno, fused)
        return result

    def fuse_greedy(self, cutoff=None, scores=None, fuse_func=None,
                    qrels=None):
        if len(self.vno_map) <= 1:
            return self

        vnos = sorted(
            self.vno_map.keys(), key=lambda x: scores[x], reverse=True)
        vnos = vnos[:cutoff]
        if not vnos:
            return TrecRunQno(self.qno, {})

        results = []
        for i in range(1, min(cutoff + 1, len(vnos) + 1)):
            fused = fuse_func(
                [self.vno_map[x].sorted_dnos() for x in vnos[:i]])
            tmp = TrecRunQno.from_doc_scores(self.qno, fused)
            s = tmp.eval(qrels)
            results.append((s[self.qno], tmp))
        best = max(results, key=itemgetter(0))
        return best[1]

    def fuse_optimal(self,
                     cutoff=None,
                     scores=None,
                     fuse_func=None,
                     qrels=None,
                     ignore_zero=False,
                     return_vnos=False):

        vnos = list(self.vno_map.keys())
        if ignore_zero and scores is not None:
            vnos = list(filter(lambda x: scores[x] > 0, vnos))

        if cutoff is not None and scores is not None:
            vnos = sorted(vnos, key=lambda x: scores[x], reverse=True)[:cutoff]

        if not vnos:
            if return_vnos:
                return TrecRunQno(self.qno, {}), []
            else:
                return TrecRunQno(self.qno, {})

        results = []
        for comb in all_combinations(vnos):
            fused = fuse_func([self.vno_map[x].sorted_dnos() for x in comb])
            fuse_run = TrecRunQno.from_doc_scores(self.qno, fused)
            s = fuse_run.eval(qrels)
            results.append((s[self.qno], fuse_run, comb))
        results = sorted(results, key=lambda x: len(x[2]))
        best = max(results, key=itemgetter(0))
        if return_vnos:
            return best[1], best[2]
        else:
            return best[1]

    def fuse_random(self, k=None, fuse_func=None):
        k = min(k, len(self.vno_map))
        selected = random.sample(self.vno_map.keys(), k=k)
        fused = fuse_func([self.vno_map[x].sorted_dnos() for x in selected])
        result = TrecRunQno.from_doc_scores(self.qno, fused)
        return result

    def __repr__(self):
        return ''.join([repr(x) for x in self.vno_map.values()])

    def __getitem__(self, key):
        return self.vno_map[key]

    def to_trec(self):
        buffer = ''
        for x in self.vno_map.values():
            buffer += x.to_trec()
        return buffer

    def __iter__(self):
        return iter(self.vno_map.values())


def parse_line(line):
    fields = line.split()
    if len(fields) == 6:
        vno, _, docid, rank, score, _ = fields
    elif len(fields) == 3:
        vno, docid, rank = fields
        score = 1.0 / int(rank)
    else:
        raise ValueError('Unknown run format')
    qno = vno.split('-')[0]
    return qno, vno, docid, rank, score


def fuse_helper(args):
    runqno, cutoff, scores, fuse_func, weighted, always_include_orig, orig_weight = args
    return runqno.fuse(
        cutoff=cutoff,
        scores=scores,
        fuse_func=fuse_func,
        weighted=weighted,
        always_include_orig=always_include_orig,
        orig_weight=orig_weight)


class TrecRun:
    def __init__(self, qno_map):
        self.qno_map = qno_map

    @staticmethod
    def from_buffer(buffer, progress=False):
        qno_map = {}
        if progress:
            buffer = tqdm(buffer, desc='Parse')
        for line in buffer:
            record = parse_line(line)
            qno_map.setdefault(record[0], []).append(record)
        kvs = list(qno_map.items())
        if progress:
            kvs = tqdm(kvs, desc='Build')
        qno_map = {k: TrecRunQno.from_records(v) for k, v in kvs}
        return TrecRun(qno_map)

    @staticmethod
    def from_file(path, progress=False):
        with open(path, 'r') as f:
            buffer = f.readlines()
        return TrecRun.from_buffer(buffer, progress)

    def ntopics(self):
        return len(self.qno_map)

    def select_original(self):
        qno_map = {k: v.original() for k, v in self.qno_map.items()}
        return TrecRun(qno_map)

    def select_variants(self):
        qno_map = {k: v.variants() for k, v in self.qno_map.items()}
        return TrecRun(qno_map)

    def select_by_vnos(self, vnos, keep_origin_if_none=False):
        qno_map = {
            k: v.select_by_vnos(vnos, keep_origin_if_none)
            for k, v in self.qno_map.items()
        }
        return TrecRun(qno_map)

    def individual(self, pos, scores):
        qno_map = {}
        for qno, runqno in self.qno_map.items():
            s = runqno.individual(pos, scores)
            qno_map[qno] = s
        return TrecRun(qno_map)

    def eval(self, qrel, reduction='none', progress_bar=False):
        itr = self.qno_map
        if progress_bar:
            itr = tqdm(itr, desc='Eval', unit='topic')

        scores = {}
        for qno in itr:
            score = self.qno_map[qno].eval(qrel)
            scores.update(score)

        if reduction == 'mean':
            return np.mean(list(scores.values()))

        return scores

    def fuse(self,
             cutoff,
             scores,
             fuse_func,
             weighted=False,
             threads=1,
             always_include_orig=False,
             orig_weight=None):
        runqnos = list(self.qno_map.values())
        qno_map = {}
        if threads == 1:
            result = map(
                fuse_helper,
                zip(runqnos, repeat(cutoff), repeat(scores), repeat(fuse_func),
                    repeat(weighted), repeat(always_include_orig),
                    repeat(orig_weight)))
            for runqno, res in zip(runqnos, result):
                qno_map[runqno.qno] = res
        else:
            score_list = [{vno: scores[vno]
                           for vno in runqno.vno_map.keys()}
                          for runqno in runqnos]
            chunksize = max(len(runqnos) // threads, 1024)
            with Pool(threads) as pool:
                result = pool.imap(
                    fuse_helper,
                    zip(runqnos, repeat(cutoff), score_list, repeat(fuse_func),
                        repeat(weighted)),
                    chunksize=chunksize)
                for runqno, res in zip(runqnos, result):
                    qno_map[runqno.qno] = res
        return TrecRun(qno_map)

    def fuse_greedy(self, cutoff=None, scores=None, fuse_func=None,
                    qrels=None):
        qno_map = {}
        for runqno in self.qno_map.values():
            r = runqno.fuse_greedy(
                cutoff=cutoff, scores=scores, fuse_func=fuse_func, qrels=qrels)
            qno_map[runqno.qno] = r
        return TrecRun(qno_map)

    def fuse_optimal(self,
                     cutoff=None,
                     scores=None,
                     fuse_func=None,
                     qrels=None,
                     ignore_zero=True,
                     return_vnos=False,
                     show_progress=True):
        qno_map = {}
        qnos, runqnos = unzip(self.qno_map.items())
        qnos, runqnos = list(qnos), list(runqnos)
        subqrels = [qrels.select_by_qno(x) for x in qnos]
        subscore = [{v: scores[v] for v in x.vnos()} for x in runqnos]
        with Pool(os.cpu_count() // 2) as pool:
            result = pool.imap_unordered(
                fuse_optimal_sp,
                zip(
                    runqnos,
                    repeat(cutoff),
                    subscore,
                    repeat(fuse_func),
                    subqrels,
                    repeat(ignore_zero),
                    repeat(return_vnos),
                ),
                chunksize=32)
            if show_progress:
                result = tqdm(result, desc='Opt', total=len(qnos))
            if return_vnos:
                result = {x[0].qno: x for x in result}
                qno_map = {x: result[x][0] for x in qnos}
                qno_vnos = {x: result[x][1] for x in qnos}
                return TrecRun(qno_map), qno_vnos
            else:
                result = {x.qno: x for x in result}
                return TrecRun(result)

    def fuse_random(self, k=None, fuse_func=None):
        qno_map = {}
        for runqno in self.qno_map.values():
            r = runqno.fuse_random(k=k, fuse_func=fuse_func)
            qno_map[runqno.qno] = r
        return TrecRun(qno_map)

    def to_trec(self):
        buffer = ''
        for runqno in self.qno_map.values():
            buffer += runqno.to_trec()
        return buffer

    def __iter__(self):
        return iter(self.qno_map.values())


def fuse_optimal_sp(args):
    runqno, cutoff, scores, fuse_func, qrels, ignore_zero, return_vnos = args
    return runqno.fuse_optimal(
        cutoff=cutoff,
        scores=scores,
        fuse_func=fuse_func,
        qrels=qrels,
        ignore_zero=ignore_zero,
        return_vnos=return_vnos)
