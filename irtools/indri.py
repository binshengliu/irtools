#!/usr/bin/env python3
import argparse
import os
import string
import subprocess
import sys
import tempfile
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

from dask.distributed import Client
from lxml import etree
from more_itertools import unzip
from tqdm import tqdm


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


class IndriRunQuery:
    def __init__(self, path, index, scheduler=None):
        self._path = path if path else "IndriRunQuery"
        self._index = index.strip()
        self._scheduler = scheduler

    def format_xml(self, qno, query, working_set=[], extra={}):
        root = etree.Element("parameters")

        node_format = etree.SubElement(root, "trecFormat")
        node_format.text = "true"

        node_index = etree.SubElement(root, "index")
        node_index.text = self._index

        for k, v in extra.items():
            etree.SubElement(root, k).text = str(v)

        node_query = etree.SubElement(root, "query")

        etree.SubElement(node_query, "number").text = qno
        etree.SubElement(node_query, "text").text = query

        for docno in working_set:
            etree.SubElement(node_query, "workingSetDocno").text = docno

        return etree.tostring(root, pretty_print=True).decode("ascii")

    def run_file(self, qno, query, working_set, extra={}):
        string = self.format_xml("0", query, working_set, extra)

        fp = tempfile.NamedTemporaryFile(mode="w")
        fp.write(string)
        fp.flush()

        indri_args = [self._path, fp.name]

        proc = subprocess.Popen(
            indri_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="ascii",
            errors="ignore",
        )
        output = []
        for line in proc.stdout:
            if "EXCEPTION" in line:
                eprint("EXCEPTION: {} {} {}".format(qno, query, line))
                raise Exception("EXCEPTION: {} {} {}".format(qno, query, line))
            line = " ".join([qno] + line.split(maxsplit=1)[1:])
            output.append(line)

        fp.close()

        output = "".join(output)
        return output

    def run_cmd(self, qno, query, extra={}):
        indri_args = [
            self._path,
            "-index=" + self._index,
            "-trecFormat=True",
            "-queryOffset=0",
            "-query={}".format(query),
        ]

        if extra:
            extra = [f"-{k}={v}" for k, v in extra.items()]
            indri_args.extend(extra)

        proc = subprocess.Popen(
            indri_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="ascii",
            errors="ignore",
        )
        output = []
        for line in proc.stdout:
            if "EXCEPTION" in line:
                eprint("EXCEPTION: {} {} {}".format(qno, query, line))
                raise Exception("EXCEPTION: {} {} {}".format(qno, query, line))
            line = " ".join([qno] + line.split(maxsplit=1)[1:])
            output.append(line)

        output = "".join(output)
        return output

    def run_single(self, packed_args):
        qno, query, working_set, extra = packed_args
        if working_set:
            output = self.run_file(qno, query, working_set, extra)
        else:
            output = self.run_cmd(qno, query, extra)

        return output

    def run_batch(
        self, qnos, queries, working_set=None, extra={}, workers=1, bar_position=None
    ):
        if working_set is None:
            working_set = repeat(None)
        else:
            assert len(working_set) == len(qnos)
        with Pool(workers) as pool:
            results = pool.imap(
                self.run_single, zip(qnos, queries, working_set, repeat(extra))
            )
            results = list(
                tqdm(
                    results,
                    total=len(qnos),
                    position=bar_position,
                    leave=False,
                    desc="indri",
                )
            )
        return results

    def run_distributed(self, qnos, queries, working_set=[], extra={}):
        """Set up a cluster first:
        dask-scheduler
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap10
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap09
        """
        client = Client(self._scheduler)
        futures = client.map(
            self.run_single, zip(qnos, queries, repeat(working_set), repeat(extra))
        )
        output = [f.result() for f in futures]
        return output


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--index", required=True, type=Path, help="Index")

    parser.add_argument(
        "--query",
        required=True,
        help="Query: csv file, or '-' for stdin. Format: qno,query",
        type=argparse.FileType("r"),
    )

    parser.add_argument(
        "--sep",
        choices=[",", "space", "none"],
        default=",",
        help="Separator between qno and query",
    )

    parser.add_argument("--scheduler", help="Scheduler")

    parser.add_argument("--count", default=1000, type=int, help="Document count")

    parser.add_argument(
        "--workers", default=os.cpu_count() // 2, type=int, help="Workers"
    )

    parser.add_argument("--output", default=sys.stdout, type=argparse.FileType("w"))

    return parser.parse_args()


def main():
    args = parse_arguments()
    content = [x.strip() for x in args.query if x.strip()]

    if args.sep != "none":
        sep = None if args.sep == "space" else args.sep
        qnos, queries = unzip(line.split(sep, maxsplit=1) for line in content)
    else:
        queries = content
        qnos = list(map(str, range(len(queries))))

    trans = str.maketrans("", "", string.punctuation)
    queries = [s.translate(trans) for s in queries]

    qnos = list(qnos)
    queries = list(queries)

    indri = IndriRunQuery(None, str(args.index.resolve()), args.scheduler)

    if args.scheduler:
        output = indri.run_distributed(qnos, queries, extra={"count": args.count})
    else:
        output = indri.run_batch(
            qnos,
            queries,
            working_set=[],
            extra={"count": args.count},
            workers=args.workers,
        )

    args.output.writelines(output)


if __name__ == "__main__":
    main()
