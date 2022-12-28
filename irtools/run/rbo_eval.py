#!/usr/bin/env python
import argparse
from functools import partial
from itertools import product
from typing import Optional

import pandas as pd
import rbo

trec_run_header = ["query_id", "Q0", "doc_id", "rank", "score", "ranker"]


def calc_rbo(x: pd.DataFrame, k: Optional[int], p: float) -> float:
    if isinstance(x.iloc[0], float):  # nan value
        return 0.0
    if isinstance(x.iloc[1], float):
        return 0.0
    rs = rbo.RankingSimilarity(x.iloc[0], x.iloc[1])
    rbo_val: float = rs.rbo(k=k, p=p, ext=True)  # top 20 get 85%
    return rbo_val


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", names=trec_run_header)
    df = df.sort_values(["query_id", "score"], ascending=[True, False])
    df = df.reset_index(drop=True)
    df = df.groupby("query_id").apply(lambda x: x["doc_id"].tolist())
    df = df.to_frame(path)
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--base")
    parser.add_argument("--run")
    parser.add_argument("-p", nargs="+", default=[0.8, 0.9, 0.95, 0.99, 1.0])
    parser.add_argument("-k", nargs="+", default=[10, 100, 1000])

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    df_base = prepare_df(args.base)
    df_run = prepare_df(args.run)

    df = pd.concat([df_base, df_run], axis=1)
    df_out = pd.DataFrame()
    for k, p in product(args.k, args.p):
        val = df.apply(partial(calc_rbo, k=k, p=p), axis=1).mean()
        df_out.loc[k, p] = val
    print(df_out.to_string())


if __name__ == '__main__':
    main()
