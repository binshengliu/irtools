# Tools

## run_query_distributed.py

Run IndriRunQuery on a [dask](https://github.com/dask/distributed)
cluster.

Specify indri path, scheduler address, and Indri parameter files.

```
run_query_distributed.py --indri /path/to/IndriRunQuery --scheduler segsresap09:8786 **.param
```

If there are too many parameter files, it's better to pass the
parameters using stdin, otherwise the shell may not function well.

```
find . -name "*.param" | run_query_distributed.py --indri /path/to/IndriRunQuery --scheduler segsresap09:8786
```

Dask workers may panic if there are too many tasks pending. A
workaround is to control the task flow manually.

```
find . -name "*.param" | run_query_distributed.py --dry | split -l 24 - splited
for f in splited*; do
  cat $f | run_query_distributed.py --indri /path/to/IndriRunQuery --scheduler segsresap09:8786
done
```

## sort_runs.py

Handy for evaluating many run files for many metrics. It exploits
multiprocessing so it's very fast.

```
sort_runs.py --measure map,P,gdeval@20 --sort-by GDEVAL-NDCG@20 cw09b.qrels a.run b.run c.run
```

## ttest_runs.py

T-test two run files for multiple measurements. It also exploits
multiprocessing.

```
ttest_runs.py --measure map,P,gdeval@20 cw09b.qrels a.run b.run
```

## filter_spam.py

Filter run files by waterloo spam score. It exploits multiprocessing
for loading the huge spam score file. Usually the bottleneck is the
read speed of the storage system.

```
usage: filter_spam.py [-h] --score [1-100] [--count COUNT]
                      [--output DIRECTORY] [--force]
                      SCORE-FILE RUN [RUN ...]
```

```
filter_spam.py --count 1000 --score 50 ClueWeb09B_Spam_Fusion.txt a.run b.run
```

## filter_oracle_run.py

Filter a run file for true relevant documents.

```
filter_oracle_run.py -n 5 -r 2 cw09b.qrels a.run > a.filtered.run
```

This command filters a.run for the first 5 documents with relevance >= 2. 
If there are not enough documents for this criteria, it will try
relevance 1 and 0 then.

## fuse_linear.py

Interpolate scores in multiple rank lists. Also exploits multiprocessing.

```
usage: fuse_linear.py [-h] (--weight WEIGHT | --sweep) RUN [RUN ...]
```

## eval_run.py

One interface for both trec_eval and gdeval.pl evaluation.

In addition to all the measure supported by `trec_eval`, it supports
customized metric `gdeval@20` style metric.

The output is a little bit tricky. For `trec_eval` NDCG, the output
name is `TREC-NDCG@20`, and for `gdeval` the output is
`GDEVAL-NDCG@20`.

## restore_ql_score.py

Convert the KL score of a run file to query likelihood score, by
multiplying the scores with its query length.

```
usage: restore_ql_score.py [-h] --param PARAM --run RUN
```

## each_server.sh

```shell
each_server.sh 'ps -ef | grep Indri | grep -v grep'
```

## set_dask_worker_nofile.sh

```shell
set_dask_worker_nofile.sh
```

