"""Logging utility functions."""

from argparse import ArgumentParser
from logging import Formatter, Logger, StreamHandler, getLogger


def create_logger(name: str, level: str) -> Logger:
    """Create a logger."""
    handler = StreamHandler()
    handler.setLevel(level.upper())
    handler.setFormatter(Formatter("%(asctime)s %(name)s %(message)s"))
    logger = getLogger(name)
    logger.setLevel(level.upper())
    logger.addHandler(handler)
    return logger


def get_log_level() -> str:
    """Get the log level from the command-line argument."""
    parser = ArgumentParser()
    parser.add_argument(
        "-l",
        "--log",
        default="warning",
    )
    args = parser.parse_args()
    return args.log
