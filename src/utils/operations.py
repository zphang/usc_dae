import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def mask(lengths):
    batch_size = len(lengths)
    max_length = max(lengths)
    if max_length == min(lengths):
        return None
    mask = torch.ByteTensor(batch_size, max_length).fill_(0)
    for i in range(batch_size):
        for j in range(lengths[i], max_length):
            mask[i, j] = 1
    return mask


def update_learning_rate(optimizer, new_learning_rate):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = new_learning_rate
    optimizer.load_state_dict(state_dict)


def mm3d2d(a, b):
    return torch.mm(a.view(-1, a.shape[-1]), b).view(
        a.shape[0], a.shape[1], b.shape[1],
    )


def simple_loss_function(loss_function):
    if loss_function == "cosine_similarity":
        def _(a, b):
            return -F.cosine_similarity(a, b).mean()

        criterion = _
    elif loss_function == "cosine_similarity+l1":
        def _(pred, true, weights, lamb):
            return (
                -F.cosine_similarity(pred, true).mean()
                + lamb * torch.abs(weights).sum(2).mean()
            )

        criterion = _
    elif loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        raise KeyError()

    return criterion


def masked_nllloss(logprobs, target, lengths, device):
    criterion = nn.NLLLoss(reduce=False)
    loss_raw = criterion(
        logprobs.view(-1, logprobs.shape[-1]),
        target.view(-1),
    )
    loss_mask = torch.ones(target.shape)
    for i, length in enumerate(lengths):
        if length < loss_mask.shape[0]:
            loss_mask[length:, i] = 0
    return (
        (loss_raw * device(
          Variable(loss_mask.view(-1)))).sum()
        / device(loss_mask).sum()
    )


def zero_loss():
    return Variable(torch.FloatTensor([0]))


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def clip_gradients(model, gradient_clipping):
    if gradient_clipping != 0.0:
        for nn_module in model.module_ls:
            torch.nn.utils.clip_grad_norm(
                nn_module.parameters(),
                gradient_clipping
            )


def shuffle(x):
    return random.sample(x, len(x))

def shuffle_2_grams(x):
    start = 1 if random.random() < 0.5 else 0
    add = [[x[0]]] if start else [[]]
    x_nest = add + [x[i:i+2] for i in range(start, len(x), 2)]
    random.shuffle(x_nest)
    return [item for sublist in x_nest for item in sublist]
