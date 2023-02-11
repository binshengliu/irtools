import logging
import os
from typing import Optional

color_fmt = "[%(asctime)s][%(name)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
log_fmt = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def configure_logging(log_level: str = "INFO", path: Optional[str] = None) -> None:
    level = logging.getLevelName(log_level)
    try:
        import colorlog

        colorlog.basicConfig(level=level, format=color_fmt)
    except Exception:
        logging.basicConfig(level=level, format=log_fmt)

    if path is not None:
        fh = logging.FileHandler(os.path.expanduser(path))
        fh.setFormatter(logging.Formatter(log_fmt))
        logging.root.addHandler(fh)
