"""
Mostly inspired from Artexte's code: https://github.com/artetxem/undreamt
"""

import torch.nn as nn


class GlobalAttention(nn.Module):
    def __init__(self, dim, alignment_function='general'):
        super(GlobalAttention, self).__init__()
        self.alignment_function = alignment_function
        if self.alignment_function == 'general':
            self.linear_align = nn.Linear(dim, dim, bias=False)
        elif self.alignment_function != 'dot':
            raise ValueError('Invalid alignment function: {0}'.format(alignment_function))
        self.softmax = nn.Softmax(dim=1)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, query, context, mask):
        """
        query: batch*dim
        context: length*batch*dim
        ans: batch*dim
        """

        context_t = context.transpose(0, 1)  # batch*length*dim

        # Compute alignment scores
        q = query if self.alignment_function == 'dot' else self.linear_align(query)
        align = context_t.bmm(q.unsqueeze(2)).squeeze(2)  # batch*length

        # Mask alignment scores
        if mask is not None:
            align.data.masked_fill_(mask, -float('inf'))

        # Compute attention from alignment scores
        attention = self.softmax(align)  # batch*length

        # Computed weighted context
        weighted_context = attention.unsqueeze(1).bmm(context_t).squeeze(1)  # batch*dim

        # Combine context and query
        return self.tanh(self.linear_context(weighted_context) + self.linear_query(query))