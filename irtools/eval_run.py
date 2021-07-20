#!/usr/bin/env python3
import os
import subprocess
import sys
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from irtools.merge_dict import merge_dict_of_dict
from irtools.trec_run import TrecRun
from tqdm import tqdm


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def gdeval_version():
    gdeval = str(Path(__file__).resolve().with_name("gdeval.pl"))
    args = [gdeval, "-version"]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stdout.decode("utf-8").strip())


def trec_eval_version():
    args = [trec_eval_path(), "--version"]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stderr.decode("utf-8").strip())


def eval_run_version():
    trec_eval_version()
    gdeval_version()


def gdeval_path():
    return str(Path(__file__).resolve().with_name("gdeval.pl"))


def trec_eval_path():
    return str(Path(__file__).resolve().with_name("trec_eval"))


def rbp_eval_path():
    return str(Path(__file__).resolve().with_name("rbp_eval"))


def trec_support():
    cmd = f"{trec_eval_path()} -h -m all_trec"
    proc = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    lines = proc.stdout.splitlines()
    start = None
    for i in enumerate(len(lines)):
        if lines[i].startswith("Individual measure documentation"):
            start = i + 1
            break
    measures = [l for l in lines[:start] if not l.startswith(" ")]
    return measures


def gdeval(measure, qrel_path, run_path, run_buffer=None):
    k = measure.split("@")[1]
    args = [gdeval_path(), "-k", k, qrel_path, run_path]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode("utf-8").splitlines()
    _, topic, ndcg, err = lines[0].split(",")
    if topic != "topic":
        raise ValueError("Unrecognizable gdeval output")

    qno_results = {}
    for line in lines[1:-1]:
        _, qno, ndcg_value, err_value = line.split(",")
        qno_results.setdefault(ndcg, {})
        qno_results.setdefault(err, {})
        qno_results[ndcg][qno] = float(ndcg_value)
        qno_results[err][qno] = float(err_value)

    return qno_results


def trec_eval(measure, qrel_path, run_path, run_buffer):
    assert run_path is None or run_buffer is None
    assert run_path is not None or run_buffer is not None

    if run_path is not None:
        args = [trec_eval_path(), "-p", "-q", "-m", measure, qrel_path, run_path]
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    else:
        args = [trec_eval_path(), "-p", "-q", "-m", measure, qrel_path, "-"]
        proc = subprocess.run(
            args,
            input=run_buffer,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    lines = proc.stdout.splitlines()

    results = {}
    for line in lines:
        measure, qno, value = line.split()
        if qno == "all":
            continue
        results.setdefault(measure, {})
        results[measure][qno] = float(value)

    return results


def rbp_eval(measure, qrel_path, run_path):
    p = measure.split("@")[1]
    args = [rbp_eval_path(), "-H", "-q", "-p", p, qrel_path, run_path]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode("utf-8").splitlines()

    qno_results = {}
    for line in lines:
        _, _, _, qno, _, _, _, value, res = line.split()
        if "nan" in value:
            continue
        if qno == "all":
            continue
        else:
            qno_results.setdefault("rbp@{}".format(p), {})
            qno_results.setdefault("rbp@{}_res".format(p), {})
            qno_results["rbp@{}".format(p)][qno] = float(value)
            qno_results["rbp@{}_res".format(p)][qno] = float(res)

    return qno_results


def trec_wrapper(zipped):
    return trec_eval(*zipped)


def gdeval_wrapper(zipped):
    return gdeval(*zipped)


def rbp_wrapper(zipped):
    return rbp_eval(*zipped)


def eval_mp(eval_wrapper, measure, qrel_path, run_path, run_buffer, progress=False):
    assert run_path is None or run_buffer is None
    assert run_path is not None or run_buffer is not None

    if run_path is not None:
        trec_run = TrecRun.from_file(run_path, progress)
    else:
        trec_run = TrecRun.from_buffer(run_buffer, progress)

    buffers = [x.to_trec() for x in trec_run]
    # tmps = [None] * len(buffers)
    if eval_wrapper.__name__.startswith("trec"):
        tmps = [None] * len(buffers)
    else:
        tmps = []
        for b in buffers:
            f = NamedTemporaryFile(mode="wt", delete=False)
            f.write(b)
            f.close()
            tmps.append(f.name)

    with Pool() as pool:
        results = pool.imap(
            eval_wrapper,
            zip(repeat(measure), repeat(qrel_path), tmps, buffers),
            chunksize=24,
        )
        if progress:
            results = tqdm(results, total=len(buffers), desc="Eval")
        results = list(results)

    for t in tmps:
        if t is not None:
            os.unlink(t)

    results = merge_dict_of_dict(results)
    agg = {m: np.mean(list(vs.values())) for m, vs in results.items()}
    return agg, results


def match_prefix(s, prefix):
    return s.startswith(prefix)


def match_true(s, prefix):
    return True


class EvalEntry(object):
    def __init__(self, match_str, match_function, eval_func):
        self.match_str = match_str
        self.match_function = match_function
        self.eval_func = eval_func


functions = [
    EvalEntry("gdeval", match_prefix, gdeval_wrapper),
    EvalEntry("rbp", match_prefix, rbp_wrapper),
    EvalEntry("", match_true, trec_wrapper),
]


def eval_run(measure, qrel_path, run_path, run_buffer, progress=False):
    """Supported measure: All names supported by trec_eval. \"gdeval\" and
    \"gdeval@k\" are also supported but are not official names.
    """
    assert run_path is None or run_buffer is None
    assert run_path is not None or run_buffer is not None

    for entry in functions:
        if entry.match_function(measure, entry.match_str):
            aggregated, qno_results = eval_mp(
                entry.eval_func, measure, qrel_path, run_path, run_buffer, progress
            )
            return aggregated, qno_results

    raise ValueError("Unrecognizable measure {}".format(measure))
