# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTM(nn.Module):
    """Encodes navigation instructions, returning hidden state context (for
    attention methods) and a decoder initial state."""

    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        padding_idx,
        dropout_ratio,
        bidirectional=False,
        num_layers=1,
    ):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        print(self.embedding)
        self.lstm = nn.LSTM(
            embedding_size,
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
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        """Expects input vocab indices as (batch, seq_len). Also requires a
        list of lengths for dynamic batching."""
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)


class EncoderLSTMOscar(nn.Module):
    """Encodes navigation instructions, returning hidden state context (for
    attention methods) and a decoder initial state."""

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
        super(EncoderLSTMOscar, self).__init__()
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
        self,
        inputs,
        lengths,
        mask=None,
        position_ids=None,
        token_type_ids=None,
    ):
        """Expects input vocab indices as (batch, seq_len). Also requires a
        list of lengths for dynamic batching."""

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


class OscarEncoder(nn.Module):
    """Encodes navigation instructions, returning hidden state context (for
    attention methods) and a decoder initial state."""

    def __init__(
        self,
        args,
        bert,
        hidden_size,
        decoder_hidden_size,
        dropout_ratio,
        bidirectional=False,
        num_layers=1,
        reverse_input=False,
    ):
        super(OscarEncoder, self).__init__()

        self.transformer_hidden_size = 768
        self.reverse_input = reverse_input
        self.dec_hidden_size = decoder_hidden_size

        self.args = args

        self.bert = bert
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            self.transformer_hidden_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=bidirectional,
        )
        self.encoder_lstm2decoder_ht = nn.Linear(
            hidden_size * self.num_directions, decoder_hidden_size
        )
        self.encoder_lstm2decoder_ct = nn.Linear(
            hidden_size * self.num_directions, decoder_hidden_size
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
        self,
        inputs,
        lengths,
        mask,
        position_ids=None,
        token_type_ids=None,
    ):
        """Expects input vocab indices as (batch, seq_len). Also requires a
        list of lengths for dynamic batching."""

        seq_max_len = mask.size(1)
        att_mask = ~mask

        outputs = self.bert(
            inputs,
            token_type_ids=token_type_ids,
            attention_mask=att_mask,
            position_ids=position_ids,
        )

        output = outputs[0]

        if self.reverse_input:
            reversed_output = torch.zeros(output.size()).to(output.device)
            reverse_idx = torch.arange(seq_max_len - 1, -1, -1)
            reversed_output[att_mask] = output[:, reverse_idx][att_mask[:, reverse_idx]]
            output = reversed_output

        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(output, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if (
            self.num_directions == 2
        ):  # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
        if self.hidden_size * self.num_directions != self.dec_hidden_size:
            c_t = self.encoder_lstm2decoder_ct(c_t)

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        ctx = self.drop(ctx)

        return (
            ctx,
            decoder_init,
            c_t,
        )  # (batch, seq_len, hidden_size*num_directions)


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, query_dim, ctx_dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None, output_tilde=True, output_prob=True):
        """Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float("inf"))
        attn = self.sm(
            attn
        )  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(
        self,
        angle_feat_size,
        embedding_size,
        hidden_size,
        dropout_ratio,
        feature_size=2048 + 4,
    ):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(angle_feat_size, self.embedding_size), nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(
        self,
        action,
        feature,
        cand_feat,
        h_0,
        prev_h1,
        c_0,
        ctx,
        ctx_mask=None,
    ):
        """
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        """
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat(
            (action_embeds, attn_feat), 1
        )  # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class AttnDecoderLSTMwithClassifier(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(
        self,
        angle_feat_size,
        embedding_size,
        hidden_size,
        dropout_ratio,
        feature_size=2048 + 4,
    ):
        super(AttnDecoderLSTMwithClassifier, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(angle_feat_size, self.embedding_size), nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)


        self.question_linear = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size // 2)),
            nn.Tanh(),
            nn.Linear(int(hidden_size // 2), 1),
        )
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(
        self,
        action,
        feature,
        cand_feat,
        prev_h1,
        c_0,
        ctx,
        ctx_mask=None,
    ):
        """
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        """

        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat(
            (action_embeds, attn_feat), 1
        )  # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        qa_logit = self.question_linear(h_tilde_drop)

        _, nav_logit = self.candidate_att_layer(
            h_tilde_drop, cand_feat, output_prob=False
        )

        return h_1, c_1, nav_logit, qa_logit, h_tilde


class SpeakerEncoder(nn.Module):
    def __init__(self, args, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.args = args
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(
            feature_size,
            self.hidden_size // self.num_directions,
            self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size // self.num_directions,
            self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=bidirectional,
        )

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        x, _ = self.attention_layer(  # Attend to the feature map
            ctx.contiguous().view(
                -1, self.hidden_size
            ),  # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(
                batch_size * max_length, -1, self.feature_size
            ),  # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerDecoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1),
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(
            0
        )  # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1)
            .expand(-1, multiplier, -1, -1)
            .contiguous()
            .view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1)
            .expand(-1, multiplier, -1)
            .contiguous()
            .view(batchXlength, -1),
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
