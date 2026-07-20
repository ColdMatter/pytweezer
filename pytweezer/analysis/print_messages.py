def print_error(message, t='error'):
    raise RuntimeError(
        "print_error() has been removed. Use a logger instead: "
        "logger.error(...), logger.warning(...), logger.info(...), logger.debug(...)."
    )


def get_timestamp(string_format='%Y/%m/%d %H:%M:%S'):
    # Kept for compatibility with callers importing this helper directly.
    import datetime

    return str(datetime.datetime.now().strftime(string_format))
