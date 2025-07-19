import logging
from pathlib import Path
from typing import Union


def setup_logger(log_path: Union[str, Path]) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger