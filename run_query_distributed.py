#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET
import sys
import signal
from dask.distributed import Client, as_completed, get_worker, wait, Queue


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def fullpath(p):
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
        type=fullpath,
        help='param file',
    )

    args = parser.parse_args()

    return args


def run_indri(args, output, queue):
    queue.put(('started', get_worker().address, output))
    import subprocess
    import os
    processes = int(len(os.sched_getaffinity(0)) * 9 / 10)
    args = (args[0], '-threads={}'.format(processes), *args[1:])
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(output, 'wb') as f:
        f.write(proc.stdout)

    queue.put(('completed', get_worker().address, output))
    return


# Idle workers are the ones with fewer tasks than its ncores/nthreads.
def idle_workers(dask_scheduler=None):
    workers = dask_scheduler.idle
    return [w.address for w in workers]


def worker_tasks(dask_scheduler=None):
    return [(w[1].address, len(w[1].processing))
            for w in dask_scheduler.workers.items()]


def list_of_workers(dask_scheduler=None):
    return dask_scheduler.workers.keys()


def run_indri_cluster(scheduler, args_output_list):
    client = Client(scheduler)
    available_workers = client.run_on_scheduler(list_of_workers)
    nworkers = len(available_workers)
    ntasks = len(args_output_list)
    eprint('{} workers:\n{}'.format(nworkers, '\n'.join(available_workers)))
    eprint('{} tasks:\n{}'.format(ntasks,
                                  '\n'.join(o for _, o in args_output_list)))

    queue = Queue()
    futures = []
    futures = client.map(run_indri, *zip(*args_output_list), [queue] * ntasks)

    def signal_handler(sig, frame):
        client.cancel(futures)
        eprint('Futures cancelled')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    sched, compl = 0, 0
    while compl < ntasks:
        status, worker, msg = queue.get()
        if status == 'started':
            sched += 1
            eprint('Scheduled {}/{} {} {}'.format(sched, ntasks, worker, msg))
        elif status == 'completed':
            compl += 1
            eprint('Completed {}/{} {} {}'.format(compl, ntasks, worker, msg))

    wait(futures)
    eprint('All tasks completed')


def check_thread_param(param_list):
    for param in param_list:
        if ET.parse(str(param)).getroot().find('threads') is not None:
            eprint('Warning: Found <threads> tag in {}. '
                   'It will be overwritten with the threads '
                   'specified by this program'.format(param))


def main():
    args = parse_args()

    check_thread_param(args.param)

    params = args.param
    runs = [p.with_suffix('.run') for p in params]
    cluster_args = [((str(args.indri), str(p)), str(r))
                    for p, r in zip(params, runs) if not r.exists()]
    run_indri_cluster(args.scheduler, cluster_args)


if __name__ == '__main__':
    main()
