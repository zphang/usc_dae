import torch
import torch.nn as nn

import src.models.generators as generators


class SimpleModel(nn.Module):
    def __init__(self, encoder, decoder, generator, learned_embeddings):
        super(SimpleModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.learned_embeddings = learned_embeddings
        self.module_ls = [
            self.encoder, self.decoder, self.generator, self.learned_embeddings
        ]

    def forward(self):
        raise NotImplementedError()

    def initialize(self, param_init=0.1):
        for param in self.parameters():
            param.data.uniform_(-param_init, param_init)
        self.encoder.special_embeddings.weight.data[0] *= 0
        self.decoder.special_embeddings.weight.data[0] *= 0

    def initialize_optimizer(self, learning_rate):
        optimizers = Optimizers()
        add_optimizer(self.encoder, [optimizers], learning_rate)
        add_optimizer(self.decoder, [optimizers], learning_rate)
        add_optimizer(self.learned_embeddings, [optimizers], learning_rate)

        # Not sure if this is necessary
        if isinstance(self.generator, generators.WrappedEmbeddingGenerator):
            generator = self.generator.embedding_generator
        else:
            generator = self.generator
        add_optimizer(generator, [optimizers], learning_rate)

        return optimizers


class TwoModel(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(SimpleModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.module_ls = [self.encoder, self.decoder, self.generator]

    def forward(self):
        raise NotImplementedError()

    def initialize(self, param_init=0.1):
        for param in self.parameters():
            param.data.uniform_(-param_init, param_init)
        self.encoder.special_embeddings.weight.data[0] *= 0
        self.decoder.special_embeddings.weight.data[0] *= 0

    def initialize_optimizer(self, learning_rate):
        optimizers = Optimizers()
        add_optimizer(self.encoder, [optimizers], learning_rate)
        add_optimizer(self.decoder, [optimizers], learning_rate)

        # Not sure if this is necessary
        if isinstance(self.generator, generators.WrappedEmbeddingGenerator):
            generator = self.generator.embedding_generator
        else:
            generator = self.generator
        add_optimizer(generator, [optimizers], learning_rate)

        return optimizers


class Optimizers:
    def __init__(self, optimizer_ls=None):
        if optimizer_ls:
            self.optimizer_ls = optimizer_ls
        else:
            self.optimizer_ls = []

    def add_optimizer(self, optimizer):
        self.optimizer_ls.append(optimizer)

    def step(self):
        for optimizer in self.optimizer_ls:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer_ls:
            optimizer.zero_grad()


def add_optimizer(nn_module, optimizers_list, lr):
    optimizer = torch.optim.Adam(nn_module.parameters(), lr=lr)
    for optimizers in optimizers_list:
        optimizers.add_optimizer(optimizer)
    return optimizer
