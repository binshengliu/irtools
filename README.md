# filter_oracle_run.py
```
python3 ../../../trec_tools/filter_oracle_run.py -h                                                                                                Fri Oct  5 11:06:25 2018
usage: filter_oracle_run.py [-h] [--number NUMBER]
                            [--min-relevance MIN_RELEVANCE]
                            [--score-type {uniform,relevance}]
                            qrels run

Filter a run file for true relevant documents.

positional arguments:
  qrels                 Documents with a score lower than the value will be
                        removed.
  run                   Run file.

optional arguments:
  -h, --help            show this help message and exit
  --number NUMBER, -n NUMBER
                        Number of relevant documents per query.
  --min-relevance MIN_RELEVANCE, -r MIN_RELEVANCE
                        Minimum relevance score.
  --score-type {uniform,relevance}, -s {uniform,relevance}
                        How to score documents.
```
