from pytweezer.logging_utils import get_logger


_TYPE_TO_LEVEL = {
    'error': 'error',
    'warning': 'warning',
    'success': 'info',
    'info': 'info',
    'bold': 'info',
    'weak': 'debug',
}


def print_error(message, t='error'):
    """Backward-compatible message API backed by the stdlib logging module."""
    logger = get_logger('pytweezer')
    level_name = _TYPE_TO_LEVEL.get(t)
    if level_name is None:
        logger.error("print_messages.py: %s is no valid key! Message = %s", t, message)
        return
    getattr(logger, level_name)(str(message))


def get_timestamp(string_format='%Y/%m/%d %H:%M:%S'):
    # Kept for compatibility with callers importing this helper directly.
    import datetime

    return str(datetime.datetime.now().strftime(string_format))
