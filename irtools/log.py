import logging
import os
import sys
from typing import Optional, TextIO


def get_logger(
    name: str,
    format: str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    level: int = logging.INFO,
    stream: Optional[TextIO] = sys.stderr,
    path: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)

    if path is not None:
        fh = logging.FileHandler(os.path.expanduser(path))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if stream is not None:
        ch = logging.StreamHandler(stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
