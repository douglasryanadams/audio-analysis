import inspect
import logging
import sys

from loguru import logger


def setup():
    logger.remove()
    logger.add(sys.stdout, level='INFO')

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
