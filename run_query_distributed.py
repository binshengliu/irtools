#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET
import sys
import signal
import time
from dask.distributed import Client, as_completed, get_worker, wait, Queue, Variable


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def fullpath(p):
    if p == '-':
        return p
    return Path(p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description='Run queries distributedly')

    parser.add_argument('--scheduler', help='Cluster scheduler address')

    parser.add_argument(
        '--indri',
        default=Path('IndriRunQuery'),
        type=fullpath,
        help='Indri path')

    parser.add_argument(
        'param',
        nargs='+',
        type=Path,
        help='param file',
    )

    args = parser.parse_args()

    return args


def run_indri(args, output, queue):
    cancel = Variable('cancel')
    if cancel.get():
        return ('canceled', get_worker().address, 0)

    start = time.time()
    import subprocess
    import os
    processes = int(len(os.sched_getaffinity(0)) * 9 / 10)
    args = (args[0], '-threads={}'.format(processes), *args[1:])

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    content = []
    for l in proc.stdout:
        content.append(l)
        if len(content) % 1000 == 0:
            if cancel.get():
                proc.kill()
                return ('killed', get_worker().address, time.time() - start)

    proc.wait()
    with open(output, 'wb') as f:
        f.writelines(content)

    return ('completed', get_worker().address, time.time() - start)


# Idle workers are the ones with fewer tasks than its ncores/nthreads.
def idle_workers(dask_scheduler=None):
    workers = dask_scheduler.idle
    return [w.address for w in workers]


def worker_tasks(dask_scheduler=None):
    return [(w[1].address, len(w[1].processing))
            for w in dask_scheduler.workers.items()]


def list_of_workers(dask_scheduler=None):
    return dask_scheduler.workers.keys()


def run_indri_cluster(scheduler, indri, params, runs):
    client = Client(scheduler)
    available_workers = client.run_on_scheduler(list_of_workers)
    nworkers = len(available_workers)
    ntasks = len(params)
    eprint('{} workers:\n{}'.format(nworkers, '\n'.join(available_workers)))
    eprint('{} tasks:\n{}'.format(ntasks, '\n'.join(str(p) for p in params)))

    queue = Queue()
    cancel = Variable('cancel')
    cancel.set(False)
    indri_args = [(str(indri.resolve()), str(p.resolve())) for p in params]
    fp_runs = [str(r.resolve()) for r in runs]
    futures = client.map(run_indri, indri_args, fp_runs, [queue] * ntasks)
    run_map = dict(zip(futures, runs))

    def signal_handler(sig, frame):
        cancel.set(True)
        eprint('Killing running tasks. This may take seconds.')
        # wait(futures)
        # sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    counter = 0
    for cf in as_completed(futures):
        counter += 1
        run = run_map[cf]
        status, addr, elap = cf.result()
        eprint('{:>3}/{:<3} {:<9} {:<27} {:.1f}s {}'.format(
            counter, ntasks, status, addr, elap, run))


def check_thread_param(param_list):
    for param in param_list:
        if ET.parse(str(param)).getroot().find('threads') is not None:
            eprint('Warning: Found <threads> tag in {}. '
                   'It will be overwritten with the threads '
                   'specified by this program'.format(param))


def main():
    args = parse_args()

    if len(args.param) == 1 and args.param[0] == '-':
        args.param = [fullpath(f.strip('\n')) for f in sys.stdin]

    # check_thread_param(args.param)

    params = [p for p in args.param if not p.with_suffix('.run').exists()]
    runs = [p.with_suffix('.run') for p in params]
    run_indri_cluster(args.scheduler, args.indri, params, runs)


if __name__ == '__main__':
    main()
