import os
import logging
import re
from datetime import datetime
import pandas as pd 


def _safe_path_part(value):
    value = str(value).strip()
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value)
    value = value.strip('._-')
    return value or None


def get_result_name(args):
    result_name = getattr(args, 'result_name', None)
    if result_name is None:
        return None
    result_name = str(result_name).strip()
    if result_name == '':
        return None
    return result_name.replace('/', '_').replace('\\', '_')


def get_result_group(args):
    result_group = getattr(args, 'result_group', None)
    if result_group is None:
        return None
    parts = []
    for part in re.split(r'[\\/]+', str(result_group)):
        safe_part = _safe_path_part(part)
        if safe_part:
            parts.append(safe_part)
    if not parts:
        return None
    return os.path.join(*parts)


def get_log_dir(args):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    base_dir = os.path.join(current_dir, 'experiments', args.model_name, args.dataset)
    result_group = get_result_group(args)
    if result_group:
        base_dir = os.path.join(base_dir, result_group)

    result_name = get_result_name(args)
    if result_name:
        log_name = 'seed_{}'.format(getattr(args, 'seed', 'run'))
        log_dir = os.path.join(base_dir, result_name, log_name)
    else:
        log_dir = os.path.join(base_dir, current_time)
    return log_dir 

def get_logger(root, name=None, debug=True):
    # when debug is true, show DEBUG and INFO in screen
    # when debug is false, show DEBUG in file and info in both screen&file
    # INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger

class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)
