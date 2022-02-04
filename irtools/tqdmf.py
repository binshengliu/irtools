import gzip
import os
import struct
from typing import Any, AnyStr, Iterator

from tqdm import tqdm


def getuncompressedsize(filename: "os.PathLike[AnyStr]") -> int:
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def tqdmf(path: "os.PathLike[AnyStr]", *args: Any, **kwargs: Any) -> Iterator[str]:
    desc = kwargs.pop("desc", str(path).rsplit("/", maxsplit=1)[-1])
    if str(path).endswith(".gz"):
        openf = gzip.open
        total = getuncompressedsize(path)
        mode = "rt"
    else:
        openf = open
        total = os.stat(path).st_size
        mode = "r"
    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        *args,
        **kwargs,
    ) as bar:
        with openf(path, mode) as f:
            for line in f:
                if bar.n + len(line) > bar.total:
                    bar.total = bar.total + 4 * (1024 ** 3)
                    bar.refresh()
                bar.update(len(line))
                yield line
