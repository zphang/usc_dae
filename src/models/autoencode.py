import numpy as np

import torch
from torch.autograd import Variable

import src.datasets.data as data
import src.utils.operations as operations
import src.models.length_control as length_control


def get_autoencode_loss(sent_batch,
                        encoder, decoder, generator,
                        corpus, dictionary, device, word_embeddings, max_length,
                        ae_noising, ae_noising_kind,
                        init_decoder_with_nli=False, nli_model=None,
                        nli_mapper_mode=None,
                        ae_add_noise_perc_per_sent_low=None,
                        ae_add_noise_perc_per_sent_high=None,
                        ae_add_noise_num_sent=None,
                        ae_add_noise_2_grams=None,
                        length_countdown_mode=None,
                        ):
    batch_size = len(sent_batch)

    if ae_noising:
        if ae_noising_kind == "additive":
            input_sent_batch = additive_noise(
                sent_batch=sent_batch,
                lengths=[len(data.tokenize(sent)) for sent in sent_batch],
                corpus=corpus,
                ae_add_noise_perc_per_sent_low=ae_add_noise_perc_per_sent_low,
                ae_add_noise_perc_per_sent_high=ae_add_noise_perc_per_sent_high,
                ae_add_noise_num_sent=ae_add_noise_num_sent,
                ae_add_noise_2_grams=ae_add_noise_2_grams,
            )
        else:
            raise NotImplementedError(f'{ae_noising_kind} noise is not implemented')

        target_ids, target_lengths, oov_dicts = dictionary.sentences2ids(
            sent_batch, sos=False, eos=True)
        input_ids, input_lengths, _ = dictionary.sentences2ids(
            input_sent_batch, sos=False, eos=True,
            given_oov_dicts=oov_dicts)
        target_ids_batch = device(Variable(torch.LongTensor(target_ids)))
        input_ids_batch = device(Variable(torch.LongTensor(input_ids)))
    else:
        target_ids, target_lengths, oov_dicts = dictionary.sentences2ids(
            sent_batch, sos=False, eos=True)
        target_ids_batch = device(Variable(torch.LongTensor(target_ids)))

        input_sent_batch = sent_batch
        input_ids, input_lengths = target_ids, target_lengths
        input_ids_batch = target_ids_batch


    hidden = device(encoder.initial_hidden(batch_size))

    # In all cases, we need the correct length target for length penalty
    # But if not in normal mode, we override auto_length_cd
    autoencode_length_countdown, autoencode_length_target = \
        length_control.get_fixed_length_data(
            lengths=target_lengths,
            max_length=max_length,
            batch_size=batch_size,
        )
    if length_countdown_mode == "noisy":
        autoencode_length_countdown, _ = \
            length_control.get_fixed_length_data(
                lengths=input_lengths,
                max_length=max_length,
                batch_size=batch_size,
            )
    elif length_countdown_mode == "none":
        rows_tensor = max(max_length, max(input_lengths))
        autoencode_length_countdown = torch.zeros((rows_tensor, batch_size, 1))
    elif length_countdown_mode != "normal":
        raise NotImplemented

    hidden, context = encoder(
        enc_input=input_ids_batch,
        lengths=input_lengths,
        word_embeddings=word_embeddings,
        hidden=hidden,
        ids_not_onehots=True,
    )

    # Auto-encoding always uses teacher-forcing
    autoencode_ids, _, _ = dictionary.sentences2ids(
        sent_batch, sos=True, eos=False, given_oov_dicts=oov_dicts)
    autoencode_ids_batch = \
        device(Variable(torch.LongTensor(autoencode_ids))).transpose(0, 1)
    autoencode_length_countdown = \
        autoencode_length_countdown[:autoencode_ids_batch.shape[0]]

    if init_decoder_with_nli:
        if nli_mapper_mode in {0, 1, 2}:
            output = decoder.nli_mapper(device(Variable(
                torch.FloatTensor(nli_model.encoder.encode(sent_batch))
            )))
        elif nli_mapper_mode == 3:
            output = device(decoder.initial_output(batch_size))
            hidden = decoder.nli_mapper(hidden, device(Variable(
                torch.FloatTensor(nli_model.encoder.encode(sent_batch))
            )))
        else:
            raise KeyError('NLI mapper mode not implemented.')
    else:
        output = device(decoder.initial_output(batch_size))

    autoencode_context_mask = device(operations.mask(input_lengths))

    autoencode_logprobs, hidden, *output = decoder(
        ids=autoencode_ids_batch,
        lengths=target_lengths,
        length_countdown=device(Variable(autoencode_length_countdown)),
        word_embeddings=word_embeddings,
        hidden=hidden,
        context=context,
        context_mask=autoencode_context_mask,
        prev_output=output,
        generator=generator,
    )

    autoencode_loss = operations.masked_nllloss(
        logprobs=autoencode_logprobs,
        target=target_ids_batch.transpose(0, 1).contiguous(),
        lengths=target_lengths,
        device=device,
    )
    return (
        autoencode_loss, autoencode_logprobs,
        oov_dicts, input_ids, target_ids, autoencode_length_target
    )


def additive_noise(sent_batch,
                   lengths,
                   corpus,
                   ae_add_noise_perc_per_sent_low,
                   ae_add_noise_perc_per_sent_high,
                   ae_add_noise_num_sent,
                   ae_add_noise_2_grams):
    assert ae_add_noise_perc_per_sent_low <= ae_add_noise_perc_per_sent_high
    batch_size = len(lengths)
    if ae_add_noise_2_grams:
        shuffler_func = operations.shuffle_2_grams
    else:
        shuffler_func = operations.shuffle
    split_sent_batch = [
        sent.split()
        for sent in sent_batch
    ]
    length_arr = np.array(lengths)
    min_add_lengths = np.floor(length_arr * ae_add_noise_perc_per_sent_low)
    max_add_lengths = np.ceil(length_arr * ae_add_noise_perc_per_sent_high)
    for s_i in range(ae_add_noise_num_sent):
        add_lengths = np.round(
            np.random.uniform(min_add_lengths, max_add_lengths)
        ).astype(int)
        next_batch = operations.shuffle(corpus.next_batch(batch_size))
        for r_i, new_sent in enumerate(next_batch):
            addition = shuffler_func(new_sent.split())[:add_lengths[r_i]]
            split_sent_batch[r_i] += addition
    noised_sent_batch = [
        " ".join(shuffler_func(sent))
        for sent in split_sent_batch
    ]
    return noised_sent_batch





def resolve_special_vocab(data_top_words_path):
    if data_top_words_path:
        with open(data_top_words_path) as f:
            special_vocab = set(f.read().split('\n'))
    else:
        special_vocab = None
    return special_vocab
