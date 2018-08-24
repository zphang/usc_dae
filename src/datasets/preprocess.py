from src.datasets.data import tokenize
import collections as col
import pandas as pd
import sys


def generate_vocabulary_file(input_file_path, output_file_path):
    word_counter = col.Counter()
    with open(input_file_path, "r") as f:
        for line in f:
            for word in tokenize(line):
                word_counter[word] += 1

    word_srs = pd.Series(word_counter)
    sorted_word_srs = word_srs.sort_values(ascending=False)

    with open(output_file_path, "w") as f:
        for word in sorted_word_srs.index[:-1]:
            f.write(word + "\n")
        f.write(sorted_word_srs.index[-1])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        generate_vocabulary_file(sys.argv[1], sys.argv[2])
