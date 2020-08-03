import os
from typing import Any, AnyStr, Iterator

from tqdm import tqdm


def tqdmf(path: "os.PathLike[AnyStr]", *args: Any, **kwargs: Any) -> Iterator[str]:
    desc = kwargs.pop("desc", str(path).rsplit("/", maxsplit=1)[-1])
    with tqdm(
        total=os.stat(path).st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        *args,
        **kwargs
    ) as bar:
        with open(path, "r") as f:
            for line in f:
                bar.update(len(line))
                yield line
