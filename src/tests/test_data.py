import numpy as np
import torch

from src.utils.misc import get_base_dir
import src.datasets.data as data


test_data_fol = get_base_dir() / "test_data"
test_data_path = test_data_fol / "test_wmt16_en.txt"
test_data_vocabulary_path = test_data_fol / "test_wmt16_en_vocabulary.txt"
test_vectors_path = test_data_fol / "test_vecs.txt"


def test_corpus():
    with open(test_data_path) as f:
        corpus = data.CorpusReader(f, max_sentence_length=100)
        batch = corpus.next_batch(3)
        assert len(batch) == 3
        batch = corpus.next_batch(30)
        assert len(batch) == 30


def test_dictionary():
    with open(test_data_path) as f:
        corpus = data.CorpusReader(f, max_sentence_length=100)
        batch = corpus.next_batch(3)
        words = sorted(list({w for s in batch for w in data.tokenize(s)}))
        dictionary = data.Dictionary(words)
        sentences, lengths = dictionary.sentences2ids(batch)
        assert np.array(sentences).shape == (3, 40)
        assert tuple(lengths) == (4, 40, 37)
        assert tuple(dictionary.sentence2ids(batch[0])) == (46, 38, 53, 49)


def test_read_embeddings_and_special_ids_and_word_ids():
    with open(test_data_vocabulary_path, "r") as f:
        word_ls = [
            line.strip()
            for line in f
        ]

    embedding, dictionary = data.read_embeddings(
        test_vectors_path,
        word_ls,
    )

    assert embedding.weight.data.shape == (9, 100)
    test_sentence = dictionary.sentence2ids("the in of to and", eos=True)
    assert tuple(test_sentence) == (5, 11, 8, 9, 10, 2)

    assert np.equal(
        data.word_ids(torch.LongTensor(test_sentence)).numpy(),
        np.array([2, 8, 5, 6, 7, 0]),
    ).all()

    assert np.equal(
        data.special_ids(torch.LongTensor(test_sentence)).numpy(),
        np.array([0, 0, 0, 0, 0, 2]),
    ).all()
