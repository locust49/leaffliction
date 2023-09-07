import logging


def configure_logger(verbose):
    logging.basicConfig(level=logging.ERROR,
                        format='[%(asctime)s] %(module)s.py - %(levelname)s : %(message)s')
    logger = logging.getLogger(__name__)

    # If verbose mode is enabled, set the logging level to INFO
    if verbose:
        logger.setLevel(logging.INFO)
    return logger