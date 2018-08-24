import os
import numpy as np
import sys
import torch

import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_nli_model(
        nli_code_path, nli_pickle_path, glove_path, word_list, verbose=True):
    assert os.path.exists(nli_code_path)
    sys.path += [nli_code_path]
    if verbose:
        print("Loading NLI..")
    nli_net = torch.load(
        nli_pickle_path,
        map_location=lambda storage, loc: storage
    )
    sys.path = sys.path[:-1]
    nli_net.encoder.set_glove_path(glove_path)
    # the argument is word_dict, but it just needs an iterator
    nli_net.encoder.word_vec = nli_net.encoder.get_glove(
        word_list, verbose=verbose)
    nli_net = nli_net.cuda()
    for param in nli_net.parameters():
        param.requires_grad = False
    if verbose:
        print("Done Loading NLI")
    return nli_net


def get_nli_loss(gs_onehot, gs_lengths, target_ids, target_lengths,
                 nli_model, encoder, word_embeddings, device):
    batch_size = gs_onehot.shape[0]
    pred_embeddings = encoder.embed_onehot(
        onehot=gs_onehot,
        word_embeddings=word_embeddings,
        include_special=False,
    )
    true_embeddings = encoder.embed_ids(
        ids=target_ids,
        word_embeddings=word_embeddings,
        include_special=False,
    )
    nli_logprobs = nli_model(
        (pred_embeddings.transpose(0, 1), np.array(gs_lengths)),
        (true_embeddings.transpose(0, 1), np.array(target_lengths)),
    )
    nli_loss = torch.nn.NLLLoss()(
        nli_logprobs,
        Variable(device(torch.LongTensor([0]*batch_size)))
    )
    return nli_loss, nli_logprobs


def resolve_nli_model(nli_code_path, nli_pickle_path, glove_path, word_list,
                      nli_loss_multiplier, init_decoder_with_nli, device):
    if nli_loss_multiplier or init_decoder_with_nli:
        return device(get_nli_model(
            nli_code_path=nli_code_path,
            nli_pickle_path=nli_pickle_path,
            glove_path=glove_path,
            word_list=word_list,
        ))
    else:
        return None


def resolve_nli_mapper(init_decoder_with_nli, nli_model, hidden_size,
                       nli_mapper_mode, rnn_type):
    if init_decoder_with_nli:
        if nli_mapper_mode == 0:
            if rnn_type == "gru":
                mapper_class = NLIMapper
            elif rnn_type == "lstm":
                mapper_class = LSTMNLIMapper
            else:
                raise KeyError(f"Rnn type {rnn_type} not handled")
            nli_mapper = mapper_class(
                nli_model.enc_lstm_dim * 2,
                hidden_size,
            )

        else:
            raise KeyError("Mapping mode not implemented/deprecated")
        return nli_mapper
    else:
        return None


class NLIMapper(torch.nn.Module):
    def __init__(self, nli_dim, hidden_size):
        super(NLIMapper, self).__init__()
        self.nli_dim = nli_dim
        self.hidden_size = hidden_size
        self.nli_output_dim = hidden_size
        self.linear = torch.nn.Linear(
            self.nli_dim + self.hidden_size,
            self.nli_output_dim,
        )

    def forward(self, encoder_hidden, infersent_input):
        infersent_repeated = infersent_input.expand(
            encoder_hidden.shape[0], *infersent_input.shape)
        nli_output = F.relu(self.linear(
            torch.cat([encoder_hidden, infersent_repeated], dim=2))
        )

        return nli_output


class LSTMNLIMapper(torch.nn.Module):
    def __init__(self, nli_dim, hidden_size):
        super(LSTMNLIMapper, self).__init__()
        self.nli_dim = nli_dim
        self.hidden_size = hidden_size
        self.h_mapper = NLIMapper(nli_dim, hidden_size)
        self.c_mapper = NLIMapper(nli_dim, hidden_size)

    def forward(self, encoder_hidden, infersent_input):
        h, c = encoder_hidden
        return (
            self.h_mapper(h, infersent_input),
            self.h_mapper(c, infersent_input),
        )
