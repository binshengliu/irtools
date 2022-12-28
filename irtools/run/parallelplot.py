import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

trec_run_header = ["query_id", "Q0", "doc_id", "rank", "score", "ranker"]


def setup_plotly() -> None:
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None


def load_run(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", names=trec_run_header)
    name = Path(path).stem
    df[name] = df["rank"]
    df = df.set_index(["query_id", "doc_id"])
    df = df[[name]]
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r", "--runs", required=True, nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--k", default=10, type=int)
    parser.add_argument("-c", "--color", default=0, type=int,
                        help="the run used to decide colors")
    parser.add_argument("--with-pdf", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_plotly()

    runs = [load_run(path) for path in args.runs]
    df = pd.concat(runs, axis=1, join="inner")
    cols = df.columns.tolist()

    for col in df.columns:
        df = df[df[col] <= args.k]
    df = df.reset_index()
    df = df.drop(columns=["query_id", "doc_id"])

    fig = px.parallel_categories(
        df,
        color=cols[args.color],
        dimensions=cols,
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.update_traces(
        dimensions=[{"categoryorder": "category ascending"} for _ in cols]
    )

    output = Path(args.output)
    fig.write_html(output.with_suffix(".html"))
    if args.with_pdf:
        fig.write_image(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
