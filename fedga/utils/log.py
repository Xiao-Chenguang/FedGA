import logging


from typing import Optional


def get_logger(
    fname: Optional[str] = None, level: Optional[int] = 30
) -> logging.Logger:
    """Get a logger with the given name and level.
    Args:
        fname optional (str): the name of the logger
        level optional (int): the level of the logger
            0: NOTSET
            10: DEBUG
            20: INFO
            30: WARNING
            40: ERROR
            50: CRITICAL

    Returns:
        logging.Logger: the logger
    """
    if fname is None:
        handlers = {logging.StreamHandler()}
    else:
        handlers = [logging.FileHandler(f"{fname}.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=level, format="%(asctime)s => %(message)s", handlers=handlers
    )
    logger = logging.getLogger(fname)
    logger.info("Logger initialized")
    logger.debug("Setting up root logger")
    return logger
