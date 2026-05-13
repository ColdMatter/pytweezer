from pytweezer.logging_utils import get_logger


def main() -> None:
    logger = get_logger("pytweezer.test")
    logger.debug("Debug message from test publisher")
    logger.info("Info message from test publisher")
    logger.warning("Warning message from test publisher")
    logger.error("Error message from test publisher")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Exception message from test publisher")
    logger.info("A really long message from test publisher " * 100)


if __name__ == "__main__":
    main()
