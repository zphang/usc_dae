"""
Get some simple stats on files for exploration
"""

import json
import numpy as np
import sys


def get_stats(path):
    lengths = []
    with open(path, "r") as f:
        for line in f.readlines():
            lengths.append(len(line.split() ))
    return {
        "average_length": np.mean(lengths),
    }


if __name__ == "__main__":
    print(json.dumps(get_stats(sys.argv[1]), indent=2))
