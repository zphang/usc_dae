import numpy as np
import torch
import torch.nn as nn


SPECIAL_SYMBOLS_ID = \
    PAD_ID, OOV_ID, EOS_ID, SOS_ID = \
    0, 1, 2, 3
SPECIAL_SYMBOLS_STR = \
    PAD_STR, OOV_STR, EOS_STR, SOS_STR = \
    "<PAD>", "<OOV>", "<EOS>", "<SOS>"
NUM_SPECIAL = 4

"""
The setup is kind of weird. It goes
0 = PAD
1 = OOV
2 = EOS
3 = SOS
4 = First Word.

The word2ids themselves have a 0th entry which is also a pad. But we'll
constantly adjust by 1 for them.
We also have numbered OOV embeddings, see HybridEmbeddings class.
"""


class Dictionary:
    def __init__(self, valid_words, num_oov):
        self.valid_words = valid_words
        self.valid_words_set = set(valid_words)
        self.num_oov = num_oov
        self.oov_list = oov_map_list(num_oov)
        words = valid_words + self.oov_list
        self.id2word = [None] + words
        self.word2id = {word: 1 + i for i, word in enumerate(words)}

    def sentence2ids(self, sentence, eos=False, sos=False, given_oov_dict=None):
        if isinstance(sentence, str):
            tokens = tokenize(sentence)
        else:
            tokens = sentence

        if given_oov_dict is not None:
            tokens = self.oov_conversion_given(tokens, given_oov_dict)
            oov_dict = given_oov_dict
        else:
            tokens, oov_dict = self.oov_conversion(tokens)

        ids = [
            NUM_SPECIAL + self.word2id[word] - 1
            if word in self.word2id else OOV_ID
            for word in tokens
        ]

        if sos:
            ids = [SOS_ID] + ids
        if eos:
            ids = ids + [EOS_ID]
        return ids, oov_dict

    def sentences2ids(self, sentences, eos=False, sos=False,
                      given_oov_dicts=None):
        ids = []
        oov_dicts = []
        for i, sentence in enumerate(sentences):
            if given_oov_dicts is not None:
                given_oov_dict = given_oov_dicts[i]
            else:
                given_oov_dict = None
            sent_ids, oov_dict = self.sentence2ids(
                sentence, eos=eos, sos=sos, given_oov_dict=given_oov_dict)
            ids.append(sent_ids)
            oov_dicts.append(oov_dict)

        lengths = [len(id_ls) for id_ls in ids]

        # Find max
        max_length = max(lengths)

        # Pad
        ids = [
            id_ls + [PAD_ID] * (max_length - len(id_ls))
            for id_ls in ids
        ]
        return ids, lengths, oov_dicts

    def ids2sentence(self, ids, oov_dict, oov_fallback=False):
        tokens = []
        for i in ids:
            if i == EOS_ID or i == PAD_ID or i == SOS_ID:
                continue
            elif i == OOV_ID:
                token = OOV_STR
            else:
                token = self.id2word[i - NUM_SPECIAL + 1]
                if is_oov(token):
                    try:
                        token = oov_dict[token]
                    except KeyError:
                        if oov_fallback:
                            token = OOV_STR
                        else:
                            raise
            tokens.append(token)
        return ' '.join(tokens)

    def ids2sentences(self, ids, oov_dicts, oov_fallback=False):
        return [
            self.ids2sentence(i, oov_dict, oov_fallback=oov_fallback)
            for i, oov_dict in zip(ids, oov_dicts)
        ]

    def rawids2sentence(self, ids, oov_dict):
        tokens = []
        for i in ids:
            if i < NUM_SPECIAL:
                token = SPECIAL_SYMBOLS_STR[i]
            else:
                token = self.id2word[i - NUM_SPECIAL + 1]
                if is_oov(token):
                    if token in oov_dict:
                        token = f"{oov_dict[token]}${remove_oov_map(token)}"
                    else:
                        token = f"{OOV_STR}${remove_oov_map(token)}"
            tokens.append(token)

        return ' '.join(tokens)

    def rawids2sentences(self, ids, oov_dicts):
        return [
            self.rawids2sentence(i, oov_dict)
            for i, oov_dict in zip(ids, oov_dicts)
        ]

    def size(self):
        return len(self.id2word) - 1

    def get_oov_dicts(self, sentences):
        return [
            self.oov_conversion(tokenize(sent))[1]
            for sent in sentences
        ]

    def oov_conversion(self, tokens):
        oov_dict = {}
        new_tokens = []
        for word in tokens:
            if word in self.valid_words_set:
                new_tokens.append(word)
            elif word in oov_dict:
                # Repeated OOV
                new_tokens.append(oov_dict[word])
            elif len(oov_dict) < len(self.oov_list):
                # OOV-#
                oov_dict[word] = oov_map(len(oov_dict))
                new_tokens.append(oov_dict[word])
            else:
                new_tokens.append(OOV_STR)
        return new_tokens, {wi: w for w, wi in oov_dict.items()}

    def oov_conversion_given(self, tokens, oov_dict):
        temp_oov_dict = {w: wi for wi, w in oov_dict.items()}
        new_tokens = []
        for word in tokens:
            if word in self.valid_words_set:
                new_tokens.append(word)
            elif word in temp_oov_dict:
                new_tokens.append(temp_oov_dict[word])
            else:
                new_tokens.append(OOV_STR)
        return new_tokens


def transform_ids(ids_tensor, start, end):
        return (ids_tensor - start + 1) * (
            (ids_tensor >= start) & (ids_tensor < end)
        ).long()


def special_ids(ids_tensor):
    return ids_tensor * (ids_tensor < NUM_SPECIAL).long()


class HybridEmbeddings(nn.Module):
    def __init__(self, fixed_embeddings, learned_embeddings):
        super(HybridEmbeddings, self).__init__()
        self.fixed_embeddings = fixed_embeddings
        self.num_fixed = self.fixed_embeddings.num_embeddings - 1

        self.learned_embeddings = learned_embeddings
        self.num_learned = self.learned_embeddings.num_embeddings - 1

    @property
    def embedding_dim(self):
        return self.fixed_embeddings.embedding_dim

    def forward(self, ids_tensor):
        fixed_ids = transform_ids(
            ids_tensor,
            start=NUM_SPECIAL,
            end=NUM_SPECIAL + self.num_fixed,
        )
        learned_ids = transform_ids(
            ids_tensor,
            start=NUM_SPECIAL + self.num_fixed,
            end=NUM_SPECIAL + self.num_fixed + self.num_learned,
        )
        embeddings = (
            self.fixed_embeddings(fixed_ids)
            + self.learned_embeddings(learned_ids)
        )
        return embeddings


class CorpusReader:
    def __init__(self, sent_file, max_sentence_length=100, loop=True):
        """ 
        Reads in fixed order. 
        No cache for now. Should be easy to upgrade later
        """
        self.sent_file = sent_file
        self.epoch = 1
        self.loop = loop
        self.max_sentence_length = max_sentence_length

    def next_batch(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            sent = self.sent_file.readline()
            if sent == "":
                self.epoch += 1
                self.sent_file.seek(0)
            else:
                sent_length = len(tokenize(sent))
                if sent_length < self.max_sentence_length:
                    batch.append(sent.strip())
        return batch

    def batch_generator(self, max_batch_size):
        while True:
            batch = []
            while len(batch) < max_batch_size:
                sent = self.sent_file.readline()
                if sent == "":
                    if batch:
                        yield batch
                    self.reset()
                    return
                else:
                    sent_length = len(tokenize(sent))
                    if sent_length < self.max_sentence_length:
                        batch.append(sent.strip())
            yield batch

    def reset(self):
        self.epoch = 0
        self.sent_file.seek(0)


class NCorpusReader:
    def __init__(self, sent_file_ls, max_sentence_length=100, loop=True):
        """
        We're going to assume the files have the same number of lines
        """
        self.sent_file_ls = sent_file_ls
        self.epoch = 1
        self.loop = loop
        self.max_sentence_length = max_sentence_length

    def next_batch(self, batch_size):
        batch_ls = [[] for _ in self.sent_file_ls]
        while len(batch_ls[0]) < batch_size:
            sent_ls = [
                sent_file.readline()
                for sent_file in self.sent_file_ls
            ]
            if any(sent == "" for sent in sent_ls):
                self.epoch += 1
                for sent_file in self.sent_file_ls:
                    sent_file.seek(0)
            else:
                for sent, batch in zip(sent_ls, batch_ls):
                    # We're just going to truncate
                    sent_tokens = tokenize(sent)
                    batch.append(" ".join(
                        sent_tokens[:self.max_sentence_length]))
        return batch_ls

    def batch_generator(self, max_batch_size):
        while True:
            batch = []
            while len(batch) < max_batch_size:
                sent = self.sent_file.readline()
                if sent == "":
                    if batch:
                        yield batch
                    self.reset()
                    return
                else:
                    sent_length = len(tokenize(sent))
                    if sent_length < self.max_sentence_length:
                        batch.append(sent.strip())
            yield batch

    def reset(self):
        self.epoch = 0
        for sent_file in self.sent_file_ls:
            sent_file.seek(0)


def read_embeddings(
        word_vector_path, vocabulary, max_read=None, num_oov=0, verbose=True):
    # Unlike undreamt, I'm assuming we have a vocabulary

    vocabulary_set = set(vocabulary)

    embeddings_vectors_ls = []
    embeddings_words_ls = []

    with open(word_vector_path, "r") as f:
        for i, line in enumerate(f):
            word, vec = line.split(' ', 1)
            if word in vocabulary_set:
                embeddings_vectors_ls.append(np.fromstring(vec, sep=' '))
                embeddings_words_ls.append(word)
                vocabulary_set.remove(word)
            if max_read is not None and i > max_read \
                    or len(embeddings_words_ls) == len(vocabulary):
                break
    learned_words = sorted(list(vocabulary_set))
    if verbose:
        print(f"Corpus words not in embeddings: {len(learned_words)}")
    learned_words_and_oov = learned_words + oov_map_list(num_oov)

    fixed_embeddings_matrix = np.array(
        [np.zeros(embeddings_vectors_ls[-1].shape)] + embeddings_vectors_ls
    )
    fixed_embeddings = nn.Embedding(
        fixed_embeddings_matrix.shape[0],
        fixed_embeddings_matrix.shape[1],
        padding_idx=0,
    )
    fixed_embeddings.weight.data.copy_(
        torch.from_numpy(fixed_embeddings_matrix))
    fixed_embeddings.weight.requires_grad = False
    learned_embeddings = nn.Embedding(
        len(learned_words_and_oov) + 1,
        fixed_embeddings_matrix.shape[1],
        padding_idx=0,
    )
    embeddings = HybridEmbeddings(
        fixed_embeddings=fixed_embeddings,
        learned_embeddings=learned_embeddings,
    )
    dictionary = Dictionary(embeddings_words_ls + learned_words, num_oov)
    dictionary.learned_words = learned_words

    return embeddings, dictionary


def load_vocabulary(vocabulary_path):
    with open(vocabulary_path, "r") as f:
        return [
            line.strip()
            for line in f
        ]


def tokenize(line):
    return line.strip().lower().split()


def resolve_embeddings_and_dictionary(
        data_vocab_path, max_vocab, vector_cache_path, vector_file_name,
        device, num_oov=0, verbose=True):
    vocabulary = load_vocabulary(data_vocab_path)[:max_vocab]
    word_embeddings, dictionary = read_embeddings(
        word_vector_path=vector_cache_path / vector_file_name,
        vocabulary=vocabulary,
        num_oov=num_oov,
        verbose=verbose,
    )
    word_embeddings = device(word_embeddings)
    return word_embeddings, dictionary


def resolve_corpus(data_path, max_sentence_length):
    return CorpusReader(
        sent_file=open(data_path, "r"),
        max_sentence_length=max_sentence_length,
    )


def raw_length(sentence):
    tokens = sentence.split()
    try:
        return tokens.index(EOS_STR)
    except ValueError:
        return len(tokens)


def oov_map(i):
    return f"<OOV-{i}>"


def remove_oov_map(string):
    return string[5:-1]


def oov_map_list(num_oov):
    return [oov_map(i) for i in range(num_oov)]


def is_oov(string):
    return string.startswith("<OOV-")
