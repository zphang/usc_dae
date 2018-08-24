"""
Some code taken from Artxexe:
https://github.com/artetxem/undreamt
"""



from src.datasets import data

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingGenerator(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(EmbeddingGenerator, self).__init__()
        self.hidden2embedding = nn.Linear(hidden_size, embedding_size)
        self.special_out = nn.Linear(embedding_size, data.NUM_SPECIAL, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, hidden, embeddings):
        emb = self.hidden2embedding(hidden)
        word_scores = torch.cat([
            F.linear(emb, embeddings.fixed_embeddings.weight[1:, :]),
            F.linear(emb, embeddings.learned_embeddings.weight[1:, :]),
        ], dim=1)
        special_scores = self.special_out(emb)
        scores = torch.cat((special_scores, word_scores), dim=1)
        return self.logsoftmax(scores)

    def output_classes(self):
        return None


class WrappedEmbeddingGenerator(nn.Module):
    def __init__(self, embedding_generator, embeddings):
        super(WrappedEmbeddingGenerator, self).__init__()
        self.embedding_generator = embedding_generator
        self.embeddings = embeddings

    def forward(self, hidden):
        return self.embedding_generator(hidden, self.embeddings)

    def output_classes(self):
        return self.embeddings.weight.data.size()[0] + data.NUM_SPECIAL - 1


class LinearGenerator(nn.Module):
    def __init__(self, hidden_size, vocabulary_size, bias=True):
        super(LinearGenerator, self).__init__()
        self.out = nn.Linear(hidden_size, data.NUM_SPECIAL + vocabulary_size, bias=bias)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, hidden):
        return self.logsoftmax(self.out(hidden))

    def output_classes(self):
        return self.out.weight.size()[0]


def resolve_generator(decoder_generator, hidden_size,
                      dictionary, word_embeddings, device):
    if decoder_generator == "linear":
        generator = device(LinearGenerator(
            hidden_size=hidden_size,
            vocabulary_size=dictionary.size(),
        ))
    elif decoder_generator == "embedding":
        embedding_generator = EmbeddingGenerator(
            hidden_size=hidden_size,
            embedding_size=word_embeddings.embedding_dim,
        )
        generator = device(WrappedEmbeddingGenerator(
            embedding_generator=embedding_generator,
            embeddings=word_embeddings,
        ))
    else:
        raise RuntimeError(f"Decoder generator {decoder_generator} not implemented")
    return generator
