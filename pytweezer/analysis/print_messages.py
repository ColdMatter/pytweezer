from termcolor import colored
import datetime


def print_error(message, t='error'):
    """
    Print an error message colored in the terminal consistently.

    :param message: str, the message to be printed
    :param t: type - one of 'error', 'warning', 'success', 'info', 'bold','weak'
    :return: None
    """

    message_formatted = get_timestamp() + '\t' + message

    if t == 'error':
        print(colored(message_formatted, 'grey', 'on_red'))
    elif t == 'warning':
        print(colored(message_formatted, 'grey', 'on_yellow'))
    elif t == 'success':
        print(colored(message_formatted, 'grey', 'on_green'))
    elif t == 'info':
        print(message_formatted)
    elif t == 'bold':
        print('\033[1m{0}\033[0m'.format(message_formatted))
    elif t == 'weak':
        print(colored(message_formatted, 'yellow'))
    else:
        print_error('print_messages.py: {0} is no valid key! Message = {1}'.format(t, message), 'error')


def get_timestamp(string_format='%Y/%m/%d %H:%M:%S'):
    return str(datetime.datetime.now().strftime(string_format))
