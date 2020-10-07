# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTM(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(
        self,
        args,
        embedding,
        embedding_size,
        hidden_size,
        dropout_ratio,
        bidirectional=False,
        num_layers=1,
    ):
        super(EncoderLSTM, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = embedding
        input_size = embedding_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=bidirectional,
        )
        self.encoder2decoder = nn.Linear(
            hidden_size * self.num_directions, hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        """ Initialize to zero cell states and hidden states."""
        batch_size = inputs.size(0)
        h0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            ),
            requires_grad=False,
        )
        c0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            ),
            requires_grad=False,
        )

        return h0.to(self.args.device), c0.to(self.args.device)

    def forward(
        self, inputs, lengths, position_ids=None, token_type_ids=None,
    ):
        """ Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. """

        embeds = self.embedding(
            inputs, position_ids=position_ids, token_type_ids=token_type_ids
        )
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if (
            self.num_directions == 2
        ):  # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        self.args.sub_out = "tanh"
        self.args.zero_init = False

        if self.args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif self.args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if self.args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return (
                ctx,
                decoder_init,
                c_t,
            )  # (batch, seq_len, hidden_size*num_directions)
            # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit
