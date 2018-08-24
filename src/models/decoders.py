import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.datasets.data as data
import src.models.attention as attention
import src.utils.operations as operations


class RNNDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_type,
                 layers=1, dropout=0, input_feeding=True):
        super(RNNDecoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(data.NUM_SPECIAL + 1,
                                               embedding_size, padding_idx=0)
        self.input_feeding = input_feeding
        self.input_size = (
            embedding_size + hidden_size
            if input_feeding
            else embedding_size
        ) + 1
        self.rnn_type = rnn_type
        if self.rnn_type == "gru":
            self.stacked_rnn = StackedGRU(
                self.input_size, hidden_size, layers=layers, dropout=dropout
            )
        elif self.rnn_type == "lstm":
            self.stacked_rnn = StackedLSTM(
                self.input_size, hidden_size, layers=layers, dropout=dropout
            )
        else:
            raise RuntimeError("RNN type not supported")
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, lengths, length_countdown,
                word_embeddings, hidden, context,
                context_mask, prev_output, generator):
        embeddings = (
            word_embeddings(ids)
            + self.special_embeddings(data.special_ids(ids))
        )
        rnn_input_batch = torch.cat((embeddings, length_countdown), dim=2)

        output = prev_output
        scores = []
        for emb in rnn_input_batch.split(1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(0), output], 1)
            else:
                input = emb.squeeze(0)
            output, hidden = self.stacked_rnn(input, hidden)
            output = self.dropout(output)
            scores.append(generator(output))
        return torch.stack(scores), hidden, output

    def forward_decode(self, hidden, context,
                       input_lengths, length_input,
                       generator, word_embeddings, device, config,
                       decoder_init=None):
        batch_size = len(input_lengths)
        length_input = length_input.unsqueeze(1)

        translations = [[] for _ in range(batch_size)]
        logprobs_ls = []
        prev_words = batch_size * [data.SOS_ID]
        pending = set(range(batch_size))
        if decoder_init is None:
            output = device(self.initial_output(batch_size))
        else:
            output = decoder_init
        context_mask = device(operations.mask(input_lengths))

        output_length = 0
        while (output_length < config.min_length or len(
                pending) > 0) and output_length < config.max_length:
            var = device(Variable(
                torch.LongTensor([prev_words]),
                requires_grad=False)
            )

            logprobs, hidden, output = self(
                ids=var,
                lengths=batch_size * [1],
                length_countdown=device(Variable(length_input[output_length])),
                word_embeddings=word_embeddings,
                hidden=hidden,
                context=context,
                context_mask=context_mask,
                prev_output=output,
                generator=generator,
            )

            prev_words = logprobs.squeeze().max(dim=1)[
                1].data.cpu().numpy().tolist()

            for i in pending.copy():
                if prev_words[i] == data.EOS_ID:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    if len(translations[i]) >= config.max_ratio * \
                            input_lengths[i]:
                        pending.discard(i)

            logprobs_ls.append(logprobs)
            output_length += 1

        all_logprobs = torch.cat(logprobs_ls)
        return all_logprobs, translations

    def initial_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size),
                        requires_grad=False)


class AttnRNNDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_type,
                 layers=1, dropout=0, input_feeding=True):
        super(AttnRNNDecoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(data.NUM_SPECIAL + 1,
                                               embedding_size, padding_idx=0)
        self.attention = attention.GlobalAttention(
            hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        self.input_size = (
            embedding_size + hidden_size
            if input_feeding
            else embedding_size
        ) + 1
        self.rnn_type = rnn_type
        if self.rnn_type == "gru":
            self.stacked_rnn = StackedGRU(
                self.input_size, hidden_size, layers=layers, dropout=dropout
            )
        elif self.rnn_type == "lstm":
            self.stacked_rnn = StackedLSTM(
                self.input_size, hidden_size, layers=layers, dropout=dropout
            )
        else:
            raise RuntimeError(f"RNN type {self.rnn_type} not implemented")
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, lengths, length_countdown,
                word_embeddings, hidden, context,
                context_mask, prev_output, generator):
        embeddings = (
            word_embeddings(ids)
            + self.special_embeddings(data.special_ids(ids))
        )
        rnn_input_batch = torch.cat((embeddings, length_countdown), dim=2)

        output = prev_output
        scores = []
        for emb in rnn_input_batch.split(1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(0), output], 1)
            else:
                input = emb.squeeze(0)
            output, hidden = self.stacked_rnn(input, hidden)
            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            scores.append(generator(output))
        return torch.stack(scores), hidden, output

    def forward_decode(self, hidden, context,
                       input_lengths, length_input,
                       generator, word_embeddings, device, config,
                       decoder_init=None):
        batch_size = len(input_lengths)
        length_input = length_input.unsqueeze(1)

        translations = [[] for _ in range(batch_size)]
        logprobs_ls = []
        prev_words = batch_size * [data.SOS_ID]
        pending = set(range(batch_size))
        if decoder_init is None:
            output = device(self.initial_output(batch_size))
        else:
            output = decoder_init
        context_mask = device(operations.mask(input_lengths))

        output_length = 0
        while (output_length < config.min_length or len(
                pending) > 0) and output_length < config.max_length:
            var = device(Variable(
                torch.LongTensor([prev_words]),
                requires_grad=False)
            )

            logprobs, hidden, output = self(
                ids=var,
                lengths=batch_size * [1],
                length_countdown=device(Variable(length_input[output_length])),
                word_embeddings=word_embeddings,
                hidden=hidden,
                context=context,
                context_mask=context_mask,
                prev_output=output,
                generator=generator,
            )

            prev_words = logprobs.squeeze().max(dim=1)[
                1].data.cpu().numpy().tolist()

            for i in pending.copy():
                if prev_words[i] == data.EOS_ID:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    if len(translations[i]) >= config.max_ratio * \
                            input_lengths[i]:
                        pending.discard(i)

            logprobs_ls.append(logprobs)
            output_length += 1

        all_logprobs = torch.cat(logprobs_ls)
        return all_logprobs, translations

    def single_decode_beam_search(self, hidden, context,
                                  input_lengths, length_input,
                                  generator, word_embeddings, device, config,
                                  beam_k, decoder_init=None):
        assert len(input_lengths) == 1

        remaining_beam_k = beam_k
        input_lengths = input_lengths * beam_k
        large_length_input = length_input.unsqueeze(1).repeat(1, 1, beam_k, 1)
        context = context.repeat(1, beam_k, 1)
        beam_batch_size = 1

        final_translations = []
        final_logprob_ls = []
        final_cumprobs = []
        translations = None
        logprobs_ls = None
        cum_logprobs = None
        prev_words = beam_batch_size * [data.SOS_ID]
        if decoder_init is None:
            output = device(self.initial_output(beam_batch_size))
        else:
            output = decoder_init
        context_mask = device(operations.mask(input_lengths * beam_k))

        output_length = 0

        while len(final_translations) < beam_k:
            var = device(Variable(
                torch.LongTensor([prev_words]),
                requires_grad=False)
            )
            logprobs, hidden, output = self(
                ids=var,
                lengths=beam_batch_size * [1],
                length_countdown=device(Variable(
                    large_length_input[output_length, :, :beam_batch_size],
                )),
                word_embeddings=word_embeddings,
                hidden=hidden,
                context=context[:, :beam_batch_size],
                context_mask=context_mask,
                prev_output=output,
                generator=generator,
            )
            top_logprobs, top_words = logprobs.squeeze(0).topk(remaining_beam_k, dim=1)
            if output_length == 0:
                beam_batch_size = remaining_beam_k
                prev_words = top_words.data.cpu().numpy().tolist()[0]
                logprob = top_logprobs.view(-1).data.cpu()
                cum_logprobs = logprob

                if self.rnn_type == "gru":
                    hidden = hidden.repeat(1, remaining_beam_k, 1),
                elif self.rnn_type == "lstm":
                    hidden = (
                        hidden[0].repeat(1, remaining_beam_k, 1),
                        hidden[1].repeat(1, remaining_beam_k, 1),
                    )
                else:
                    raise RuntimeError()
                output = output.repeat(remaining_beam_k, 1)
                translations = [[word] for word in prev_words]
                logprobs_ls = [[log_prob] for log_prob in logprob]
            else:
                candidate_cum_logprobs = top_logprobs.data.cpu() + cum_logprobs.view(-1, 1)
                flat_cum_probs = candidate_cum_logprobs.view(-1)
                beam_index = np.tile(np.arange(remaining_beam_k), (remaining_beam_k, 1)).T
                top_cum_logprobs, top_indices = flat_cum_probs.topk(remaining_beam_k)
                beam_chosen = beam_index.reshape(-1)[top_indices.numpy()]

                new_translations = []
                new_logprobs_ls = []
                new_cum_logprobs = []
                beam_pos = []
                prev_words = []
                for i in range(remaining_beam_k):
                    index = top_indices[i]
                    beam_index = beam_chosen[i]
                    word_id = top_words.view(-1)[index].data[0]
                    if word_id == data.EOS_ID or \
                                            output_length + 1 > config.max_ratio * input_lengths[i]:
                        remaining_beam_k -= 1
                        final_translations.append(translations[i])
                        final_logprob_ls.append(logprobs_ls[i])
                        final_cumprobs.append(top_cum_logprobs[i])
                    else:
                        new_translations.append(translations[beam_index] + [word_id])
                        new_logprobs_ls.append(logprobs_ls[beam_index] + [top_logprobs.view(-1)[index].data[0]])
                        new_cum_logprobs.append(top_cum_logprobs[i])
                        beam_pos.append(beam_index)
                        prev_words.append(word_id)

                translations = new_translations
                logprobs_ls = new_logprobs_ls
                cum_logprobs = torch.Tensor(new_cum_logprobs)

                beam_batch_size = remaining_beam_k
                if beam_batch_size:
                    if self.rnn_type == "gru":
                        hidden = torch.stack([hidden[:, pos] for pos in beam_pos], dim=1)
                    elif self.rnn_type == "lstm":
                        hidden = (
                            torch.stack([hidden[0][:, pos] for pos in beam_pos], dim=1),
                            torch.stack([hidden[1][:, pos] for pos in beam_pos], dim=1),
                        )
                    else:
                        raise RuntimeError()
                    output = torch.stack([output[pos] for pos in beam_pos], dim=0)
            output_length += 1

        sorted_index = np.argsort(final_cumprobs)[::-1]
        final_translations = [final_translations[i] for i in sorted_index]
        final_logprob_ls = [final_logprob_ls[i] for i in sorted_index]

        return final_logprob_ls, final_translations

    def forward_decode_beam_search(self, hidden, context,
                                   input_lengths, length_input,
                                   generator, word_embeddings, device, config,
                                   beam_k, decoder_init=None):
        batch_size = len(input_lengths)
        logprob_ls =[]
        translations = []
        for i in range(batch_size):
            if self.rnn_type == "gru":
                sub_hidden = hidden[:, i:i + 1, :]
            elif self.rnn_type == "lstm":
                sub_hidden = (
                    hidden[0][:, i:i+1, :],
                    hidden[1][:, i:i+1, :],
                )
            else:
                raise RuntimeError()
            sub_context = context[:, i:i+1, :]
            sub_input_lengths = [input_lengths[i]]
            sub_length_input = length_input[:, i:i+1, :]
            if decoder_init is not None:
                sub_decoder_init = decoder_init[i: i+1, :]
            else:
                sub_decoder_init = None

            k_logprob_ls, k_translations = self.single_decode_beam_search(
                hidden=sub_hidden,
                context=sub_context,
                input_lengths=sub_input_lengths,
                length_input=sub_length_input,
                generator=generator,
                word_embeddings=word_embeddings,
                device=device,
                config=config,
                beam_k=beam_k,
                decoder_init=sub_decoder_init,
            )
            translations.append(k_translations[0])
            logprob_ls.append(k_logprob_ls[0])

        return logprob_ls, translations

    def initial_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size),
                        requires_grad=False)


# Based on OpenNMT-py
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, input_size, hidden_size, layers, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()

        for i in range(layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
