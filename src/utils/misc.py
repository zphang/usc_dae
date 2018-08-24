import os
import pathlib


def get_base_dir():
    return (
        pathlib.Path(os.path.realpath(os.path.dirname(__file__))) / ".." / ".."
    ).resolve()


def datetime_format(timestamp):
    return f"{timestamp:%Y.%m.%d.%H.%M.%S}." + f"{timestamp:%f}"[:3]
