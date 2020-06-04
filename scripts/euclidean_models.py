import argparse
from itertools import combinations

import pandas as pd
import torch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--names")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    names = args.names.split(",") if args.names else args.checkpoints

    keys = None
    params = []
    import pdb

    pdb.set_trace()
    for name, ckpt in zip(names, args.checkpoints):
        state_dict = torch.load(ckpt)["state_dict"]  # type: ignore
        if keys is None:
            keys = state_dict.keys()
        assert keys is not None
        model_param = torch.cat([state_dict[k].reshape(-1) for k in keys])
        params.append((name, model_param))
    df = pd.DataFrame()
    for (name1, m1), (name2, m2) in combinations(params, 2):
        dist = torch.dist(m1, m2).item()
        df.loc[name1, name2] = dist
    print(df.to_string())


if __name__ == "__main__":
    main()
