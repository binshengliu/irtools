#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET
import sys
import signal
import time
from dask.distributed import Client, as_completed, get_worker, Variable, get_client
import logging


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

    parser.add_argument('--dry', action='store_true')

    parser.add_argument('--log', type=Path, help='Log path')

    parser.add_argument(
        'param',
        nargs='*',
        type=Path,
        help='Param file, none for reading from stdin',
    )

    args = parser.parse_args()

    return args


def run_indri(args, output):
    cancel = Variable('cancel', get_client())
    if cancel.get():
        return ('canceled', get_worker().address, 0)

    start = time.time()
    from subprocess import Popen, PIPE
    import os
    processes = len(os.sched_getaffinity(0)) - 1
    args = (args[0], '-threads={}'.format(processes), *args[1:])

    with Popen(args, stdout=PIPE, stderr=PIPE) as proc:
        content = []
        for l in proc.stdout:
            content.append(l)
            if len(content) % 1000 != 0:
                continue
            if cancel.get():
                proc.kill()
                return ('killed', get_worker().address, time.time() - start)

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
    ntasks = len(params)
    batchsize = len(available_workers) * 6
    for w in available_workers:
        logging.info('{}'.format(w))
    logging.info('{} tasks in total'.format(len(params)))
    logging.info('{} workers in total'.format(len(available_workers)))
    logging.info('{} tasks per round'.format(batchsize))

    cancel = Variable('cancel', client)
    cancel.set(False)

    def signal_handler(sig, frame):
        cancel.set(True)
        logging.info(
            'CTRL-C received. It may take seconds to kill running tasks.')

    signal.signal(signal.SIGINT, signal_handler)

    indri_args = [(str(indri.resolve()), str(p.resolve())) for p in params]
    fp_runs = [str(r.resolve()) for r in runs]
    submitted = 0
    completed = 0
    while submitted < ntasks:
        futures = client.map(
            run_indri,
            indri_args[submitted:submitted + batchsize],
            fp_runs[submitted:submitted + batchsize],
            key=fp_runs[submitted:submitted + batchsize],
        )
        logging.info('Submitted {} tasks'.format(len(futures)))
        run_map = dict(zip(futures, runs[submitted:submitted + batchsize]))
        for cf in as_completed(futures):
            completed += 1
            run = run_map[cf]
            status, addr, elap = cf.result()
            logging.info('{:>3}/{:<3} {:<9} {:<27} {:4.1f}s {}'.format(
                completed, ntasks, status, addr, elap, run))

        submitted += batchsize


def check_thread_param(param_list):
    for param in param_list:
        if ET.parse(str(param)).getroot().find('threads') is not None:
            logging.warn('Warning: Found <threads> tag in {}. '
                         'It will be overwritten with the threads '
                         'specified by this program'.format(param))


def setup_logging(log_file=None):
    handlers = []
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    handlers.append(stream_handler)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        handlers=handlers,
        level=logging.DEBUG)


def main():
    args = parse_args()

    setup_logging(args.log)

    if not args.param:
        args.param = [Path(f.strip('\n')) for f in sys.stdin]
    # check_thread_param(args.param)

    params = [p for p in args.param if not p.with_suffix('.run').exists()]
    if args.dry:
        for p in params:
            print('{}'.format(str(p)))
        return
    runs = [p.with_suffix('.run') for p in params]

    for p in params:
        logging.info('{}'.format(str(p)))
    run_indri_cluster(args.scheduler, args.indri, params, runs)


if __name__ == '__main__':
    main()
