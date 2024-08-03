import os
import logging

def get_logger(level='DEBUG'):
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def get_dir(*sub_dirs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def get_path(*dirs, filename):
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path