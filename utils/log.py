"""
Log utils. Used to manage the output format of AutoGL
"""

import logging
import os


def get_logger(dir,name):
    """
    Get the logger of given name

    Parameters
    ----------
    name: str
        The name of logger

    Returns
    -------
    logger: Logger
        The logger generated
    """
    if not os.path.exists(dir):
        print('Creating log folder ' + dir)
        os.mkdir(dir)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(dir,name+'.log'),
                        filemode='w')
    return logging.getLogger(name)
