#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET
import sys
from dask.distributed import Client, as_completed, get_worker, wait


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


def run_indri(args, output):
    import subprocess
    import os
    processes = int(len(os.sched_getaffinity(0)) * 9 / 10)
    args = (args[0], '-threads={}'.format(processes), *args[1:])
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(output, 'wb') as f:
        f.write(proc.stdout)

    return get_worker().address


def empty_worker(dask_scheduler=None):
    workers = dask_scheduler.workers
    return next(w for w, ws in workers.items() if not ws.processing)


def worker_tasks(dask_scheduler=None):
    return [(w[1].address, len(w[1].processing))
            for w in dask_scheduler.workers.items()]


def number_of_workers(dask_scheduler=None):
    return len(dask_scheduler.workers)


def list_of_workers(dask_scheduler=None):
    return dask_scheduler.workers.keys()


def run_indri_cluster(scheduler, args_output_list):
    client = Client(scheduler)
    workers = client.run_on_scheduler(list_of_workers)
    nworkers = len(workers)
    ntasks = len(args_output_list)
    eprint('{} workers: {}'.format(nworkers, ' '.join(workers)))
    eprint('{} tasks'.format(ntasks))

    futures = []
    scheduled_count = 0
    for args, worker in zip(args_output_list[:nworkers], workers):
        new_future = client.submit(
            run_indri, key=args[1], *args, workers=[worker])
        futures.append(new_future)
        scheduled_count += 1
        eprint('Schedule {}/{} {} {}'.format(scheduled_count, ntasks, worker,
                                             args[1]))

    args_output_list = args_output_list[nworkers:]
    completed_count = 0
    ac = as_completed(futures)
    for completed in ac:
        completed_count += 1
        eprint('Complete {}/{} {} {}'.format(completed_count, ntasks,
                                             completed.result(),
                                             completed.key))
        worker = client.run_on_scheduler(empty_worker)
        if worker and args_output_list:
            args = args_output_list.pop()
            new_future = client.submit(
                run_indri, key=args[1], *args, workers=[worker])
            ac.add(new_future)
            scheduled_count += 1
            eprint('Schedule {}/{} {} {}'.format(scheduled_count, ntasks,
                                                 worker, args[1]))

    wait(futures)
    eprint('All tasks completed')


def check_thread_param(param_list):
    for param in param_list:
        if ET.parse(str(param)).getroot().find('threads') is not None:
            eprint('Warning: Found <threads> tag in {}. '
                   'The threads specified by this program '
                   'may not work.'.format(param))


def main():
    args = parse_args()

    check_thread_param(args.param)

    cluster_args = [((str(args.indri), str(param)),
                     str(param.with_suffix('.run'))) for param in args.param]
    run_indri_cluster(args.scheduler, cluster_args)


if __name__ == '__main__':
    main()
