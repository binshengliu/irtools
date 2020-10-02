import argparse
from typing import List

import numpy as np
import pandas as pd
from irtools.evalfile import TrecEval
from irtools.seaborn_setup import seaborn_setup
from more_itertools import always_iterable


def prepare_eval(args: argparse.Namespace) -> List[pd.DataFrame]:
    seaborn_setup()
    if hasattr(args, "seed"):
        np.random.seed(args.seed)
    dfs = [TrecEval(x).to_frame() for x in always_iterable(args.eval)]
    args.names = (
        args.names.split(",") if args.names else [f"Sys{i}" for i in range(len(dfs))]
    )
    assert len(args.names) == len(args.eval)

    if args.metric:
        dfs = [x.loc[:, args.metric] for x in dfs]

    if hasattr(args, "sample") and args.sample is not None:
        if args.sample >= 1:
            dfs[0] = dfs[0].sample(n=int(args.sample), random_state=args.seed)
        else:
            dfs[0] = dfs[0].sample(frac=args.sample, random_state=args.seed)
        dfs = [x.loc[dfs[0].index] for x in dfs]

    sorted_metrics = sorted(dfs[0].columns)
    dfs = [df[sorted_metrics] for df in dfs]
    return dfs
