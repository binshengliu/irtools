from concurrent.futures import ProcessPoolExecutor as Pool
from subword_nmt import apply_bpe
from pathlib import Path
import argparse


def create_subword_bpe(codes):
    bpe_parser = apply_bpe.create_parser()
    bpe_args = bpe_parser.parse_args(['--codes', str(codes)])
    bpe = apply_bpe.BPE(bpe_args.codes)
    return bpe


def applybpe(codes, outdir, files):
    outdir = Path(outdir)
    bpe = create_subword_bpe(codes)

    with Pool() as pool:
        for path in files:
            path = Path(path)
            outfile = outdir.joinpath(path.name)
            with open(path, 'r') as fp:
                bped = pool.map(bpe.process_line, fp, chunksize=1024)
                outfile.write_text(''.join(bped))


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('codes')
    parser.add_argument('files')
    return parser.parse_args()


def main():
    args = parse_arguments()
    applybpe(args.codes, args.outdir, args.files)


if __name__ == '__main__':
    main()
