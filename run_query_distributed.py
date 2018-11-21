#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET
import sys
import signal
import time
from dask.distributed import Client, as_completed, get_worker, Variable, get_client
import logging
import os


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
    from subprocess import Popen, PIPE
    import os

    cancel = Variable('cancel', get_client())
    if cancel.get():
        return ('canceled', get_worker().address, 0, os.getloadavg())

    start = time.time()
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
                return ('killed', get_worker().address, time.time() - start,
                        os.getloadavg())

    with open(output, 'wb') as f:
        f.writelines(content)

    return ('completed', get_worker().address, time.time() - start,
            os.getloadavg())


# Idle workers are the ones with fewer tasks than its ncores/nthreads.
def idle_workers(dask_scheduler=None):
    workers = dask_scheduler.idle
    return [w.address for w in workers]


def worker_tasks(dask_scheduler=None):
    return [(w[1].address, len(w[1].processing))
            for w in dask_scheduler.workers.items()]


def list_of_workers(dask_scheduler=None):
    return dask_scheduler.workers.keys()


def get_load_info():
    processes = len(os.sched_getaffinity(0))
    return (get_worker().address, processes, os.getloadavg())


def get_worker_load(client):
    worker_info = []
    all_workers = client.run_on_scheduler(list_of_workers)
    for worker in all_workers:
        future = client.submit(get_load_info, workers=[worker], pure=False)
        w, processes, (one, five, fifteen) = future.result()
        worker_info.append((w, processes, one, five, fifteen))

    return worker_info


def find_low_loadavg_workers(client, busy_ratio):
    """
    Find workers that are both idle and have low loadavg.
      idle and high loadavg: others are using
      idel and low loadavg: nobody is using
      busy and high loadavg: I'm contending with others
      busy and low loadavg: I'm using
    """
    worker_info = []
    all_workers = client.run_on_scheduler(idle_workers)
    if not all_workers:
        return []

    for worker in all_workers:
        future = client.submit(get_load_info, workers=[worker], pure=False)
        w, processes, (one, five, fifteen) = future.result()
        worker_info.append((w, processes, one, five, fifteen))

    available = [(w, processes, one, five, fifteen)
                 for w, processes, one, five, fifteen in worker_info
                 if one < processes * busy_ratio]
    if not available:
        available = [max(worker_info, key=lambda i: i[1] - i[2])]
    return available


def schedule_loop(client, ntasks, cancel, runs, indri_args, fp_runs):
    current = 0
    worker_loads = find_low_loadavg_workers(client, 0.50)[:len(runs) - current]
    run_map = {}
    for worker, *loadavg in worker_loads:
        f = client.submit(
            run_indri,
            indri_args[current],
            fp_runs[current],
            key=fp_runs[current],
            workers=[worker])
        run_map[f] = runs[current]
        current += 1

    ac = as_completed(run_map)
    for i, cf in enumerate(ac):
        run = run_map[cf]
        status, addr, elap, loadavg = cf.result()
        logging.info('{:>3}/{:<3} {:<9} {:<27}{:<22} {:4.1f}s {}'.format(
            i + 1, ntasks, status, addr, str(loadavg), elap, run))

        if cancel.get() or current >= len(runs):
            continue

        worker_loads = find_low_loadavg_workers(client,
                                                0.50)[:len(runs) - current]
        for worker, *_ in worker_loads:
            f = client.submit(
                run_indri,
                indri_args[current],
                fp_runs[current],
                key=fp_runs[current],
                workers=[worker])
            run_map[f] = runs[current]
            ac.add(f)
            current += 1


def run_indri_cluster(scheduler, indri, params, runs):
    client = Client(scheduler)
    available_workers = get_worker_load(client)
    ntasks = len(params)
    for w in available_workers:
        logging.info('{:<27} {} {:>5.2f} {:>5.2f} {:>5.2f}'.format(*w))
    logging.info('{} tasks in total'.format(len(params)))
    logging.info('{} workers in total'.format(len(available_workers)))

    cancel = Variable('cancel', client)
    cancel.set(False)

    def signal_handler(sig, frame):
        cancel.set(True)
        logging.info(
            'CTRL-C received. It may take seconds to kill running tasks.')

    signal.signal(signal.SIGINT, signal_handler)

    indri_args = [(str(indri.resolve()), str(p.resolve())) for p in params]
    fp_runs = [str(r.resolve()) for r in runs]
    schedule_loop(client, ntasks, cancel, runs, indri_args, fp_runs)


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
