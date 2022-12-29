import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--control", required=True)
    parser.add_argument("--experimental", required=True)
    parser.add_argument("--outdir")
    parser.add_argument("--sep", default=" ")

    return parser.parse_args()


def main() -> None:
    sns.set_theme("paper", "darkgrid", font="Linux Biolinum O")

    args = parse_arguments()
    df_c = pd.read_csv(
        args.control, sep=args.sep, names=["measure", "query_id", "control"]
    )
    df_e = pd.read_csv(
        args.experimental, sep=args.sep, names=["measure", "query_id", "experimental"]
    )

    df = pd.merge(df_c, df_e)

    outdir = Path(args.outdir)
    for measure, dft in df.groupby("measure"):
        minval = min(dft[["control", "experimental"]].min())
        maxval = min(dft[["control", "experimental"]].max())

        g = sns.jointplot(
            x="control",
            y="experimental",
            data=dft,
            joint_kws={"x_jitter": 1, "y_jitter": 1},
        )
        sns.lineplot(
            x=[minval, maxval], y=[minval, maxval], alpha=0.5, lw=0.5, ax=g.ax_joint
        )
        above = (dft["experimental"] > dft["control"]).sum()
        below = (dft["experimental"] < dft["control"]).sum()
        same = (dft["experimental"] == dft["control"]).sum()
        text = f"Above: {above}\nBelow: {below}\nSame: {same}"
        g.ax_joint.annotate(
            text, (0.5, 0.9), xycoords="axes fraction", ha="center", va="center"
        )

        if isinstance(args.outdir, str):
            g.savefig(outdir.joinpath(f"joint-{measure}.pdf"))
            print(outdir.joinpath(f"joint-{measure}.pdf"))


if __name__ == "__main__":
    main()
