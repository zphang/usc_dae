import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import src.datasets.data as data


def get_length_penalty(
        log_probs, length_target, output_length,
        device):
    length_criterion = nn.BCEWithLogitsLoss()
    length_penalty = length_criterion(
        log_probs[:, :, data.EOS_ID],
        Variable(device(length_target[:output_length])),
    )
    return length_penalty


def sample_desired_lengths(original_lengths, low2=-1, high2=0.1):
    """
    low2 and high2 are exponents of 2.
    """
    return np.floor(
        np.array(original_lengths) * 2 ** (
            np.random.uniform(size=len(original_lengths), low=low2, high=high2)
        )
    )


def get_length_data_from_desired(desired_lengths, max_length, batch_size):
    length_countdown_batch = np.tile(
        np.arange(0, -max_length, -1),
        reps=(batch_size, 1)
    ) + desired_lengths.reshape(batch_size, 1)
    return (
        length_countdown_batch.astype(int),
        (length_countdown_batch == 0).astype(int),
    )


def sample_length_data(original_lengths, length_low2, length_high2, max_length,
                       batch_size):
    desired_lengths = sample_desired_lengths(
        original_lengths=original_lengths,
        low2=length_low2,
        high2=length_high2,
    )
    length_countdown, length_target = get_length_data_from_desired(
        desired_lengths=desired_lengths,
        max_length=max_length,
        batch_size=batch_size,
    )
    return (
        torch.FloatTensor(length_countdown).transpose(0, 1).unsqueeze(2),
        torch.FloatTensor(length_target).transpose(0, 1),
        desired_lengths,
    )


def get_fixed_length_data(lengths, max_length, batch_size):
    length_countdown, length_target = get_length_data_from_desired(
        desired_lengths=np.array(lengths),
        max_length=max_length,
        batch_size=batch_size,
    )
    return (
        torch.FloatTensor(length_countdown).transpose(0, 1).unsqueeze(2),
        torch.FloatTensor(length_target).transpose(0, 1),
    )