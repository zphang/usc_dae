import contextlib
import os
import sys

import src.utils.misc as misc


class AbstractLogger:
    def write(self):
        raise NotImplementedError

    def write_line(self):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def maybe_print(self):
        raise NotImplementedError


class SimpleFileLogger:
    def __init__(self, handle, also_print=False):
        self.handle = handle
        self.also_print = also_print

    def write(self, string):
        self.handle.write(string)
        self.maybe_print(string, sep="")

    def write_line(self, string):
        self.handle.write(string + "\n")
        self.maybe_print(string)

    def flush(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def maybe_print(self, string, sep="\n"):
        if self.also_print:
            print(string, sep=sep)


class EmptyLogger:
    def write(self):
        pass

    def write_line(self):
        pass

    def flush(self):
        pass

    def maybe_print(self):
        pass


class PrintLogger:
    def write(self, string):
        print(string, sep="")

    def write_line(self, string):
        print(string)

    def flush(self):
        sys.stdout.flush()

    def maybe_print(self, string, sep="\n"):
        print(string, sep=sep)


@contextlib.contextmanager
def log_context(log_file_path, also_print=False):
    with open(log_file_path, "a") as f:
        logger = SimpleFileLogger(handle=f, also_print=also_print)
        yield logger


def get_date_log_path(log_folder_path, timestamp):
    log_file_path = log_folder_path / (misc.datetime_format(timestamp) + ".log")
    return log_file_path


def get_named_date_log_path(log_folder_path, run_name, timestamp):
    log_file_path = (
        log_folder_path /
        f"{run_name}__{misc.datetime_format(timestamp)}.log"
    )
    return log_file_path


empty_logger = EmptyLogger()
print_logger = PrintLogger()
