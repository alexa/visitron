# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from oscar.modeling_bert import CaptionBertEncoder

from transformers.pytorch_transformers.modeling_bert import (
    BertEmbeddings,
    BertLayerNorm,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
)

logger = logging.getLogger(__name__)


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
            attn.masked_fill_(mask, -float("inf"))
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


class ImageBertForSequenceClassificationwithAction(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config):
        super(ImageBertForSequenceClassificationwithAction, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModelwithAction(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.candidate_att_layer = SoftDotAttention(self.config.hidden_size, 2048)

        self.apply(self.init_weights)

    def resize_embeddings(self, embedding_size_dict):
        for embedding_type, new_embedding_size in embedding_size_dict.items():
            assert embedding_type in [
                "word_embeddings",
                "position_embeddings",
                "token_type_embeddings",
            ]
            self.bert.resize_specific_embeddings(embedding_type, new_embedding_size)
            logger.info(f"Resized {embedding_type} to {new_embedding_size}")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        img_feats=None,
        action_feats=None,
        candidate_feats=None,
        text_only=False,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            img_feats=img_feats,
            action_feats=action_feats,
        )
        if text_only:
            return outputs
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        _, logits = self.candidate_att_layer(
            pooled_output, candidate_feats, output_prob=False
        )

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        return outputs


class NextActionPrediction(nn.Module):
    """
    N-class classification model
    """

    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(
            self.linear(x)
        )  # the 0-35 is the vision, 36th is the CLS token


class BertImgModelwithLocationEmbeds(BertPreTrainedModel):
    """Expand from BertModel to handle image region features as input"""

    def __init__(self, config):
        super(BertImgModelwithLocationEmbeds, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info("BertImgModel Image Dimension: {}".format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, "use_img_layernorm"):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.location_embeds = nn.Linear(128, self.config.hidden_size, bias=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.img_layer_norm_eps
            )

        self.apply(self.init_weights)

    def resize_specific_embeddings(self, embedding_type, new_num_tokens):
        """ embedding_type: word_embeddings or position_embeddings or token_type_embeddings """
        old_embeddings = getattr(self.embeddings, embedding_type)
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        setattr(self.embeddings, embedding_type, new_embeddings)
        return getattr(self.embeddings, embedding_type)

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        img_feats=None,
        img_location_embeddings=None,
        encoder_history_states=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        if encoder_history_states:
            assert (
                img_feats is None
            ), "Cannot take image features while using encoder history states"

        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)
            img_location_embeds_output = self.location_embeds(img_location_embeddings)
            img_embedding_output = img_embedding_output + img_location_embeds_output
            if self.use_img_layernorm:
                img_embedding_output = self.LayerNorm(img_embedding_output)

            # add dropout on image embedding
            img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            encoder_history_states=encoder_history_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs


class PreTrainOscar(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config):
        super(PreTrainOscar, self).__init__(config)
        self.config = config
        self.bert = BertImgModelwithLocationEmbeds(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.next_action = NextActionPrediction(
            self.config.hidden_size, self.config.action_space
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)
        self.token_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.detector_classes),
            nn.Softmax(dim=-1),
        )

        self.apply(self.init_weights)

        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(
            self.mlmhead.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def resize_embeddings(self, embedding_size_dict):
        for embedding_type, new_embedding_size in embedding_size_dict.items():
            assert embedding_type in [
                "word_embeddings",
                "position_embeddings",
                "token_type_embeddings",
            ]
            self.bert.resize_specific_embeddings(embedding_type, new_embedding_size)
            logger.info(f"Resized {embedding_type} to {new_embedding_size}")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        token_labels=None,
        position_ids=None,
        head_mask=None,
        img_feats=None,
        img_location_embeddings=None,
        next_action=None,
        text_only=False,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            img_feats=img_feats,
            img_location_embeddings=img_location_embeddings,
        )

        if text_only:
            return outputs

        cls_part = outputs[1]
        lang_part = outputs[0]

        prediction_scores = self.mlmhead(lang_part)

        token_loss = 0
        if token_labels is not None:
            token_prediction = self.token_head(lang_part)
            token_loss = self.criterion(
                token_prediction.view(-1, self.config.detector_classes),
                token_labels.view(-1),
            )

        mask_loss = self.criterion(
            prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
        )

        action_scores = self.next_action(cls_part)

        next_loss = 0
        if next_action is not None:
            next_loss = self.criterion(action_scores, next_action)
        loss = mask_loss + next_loss + token_loss

        predicted_action = torch.argmax(action_scores, dim=1)
        predicted_words = torch.argmax(prediction_scores, dim=2)
        token_prediction = torch.argmax(token_prediction, dim=2)

        predicted_words[labels == -1] = -1

        ignored_words_no = torch.sum(labels == -1)
        words_left = (labels.shape[0] * labels.shape[1]) - ignored_words_no
        words_left = words_left.type(torch.float)

        words_accuracy = (
            torch.sum(predicted_words == labels) - ignored_words_no
        ) / words_left

        if next_action is not None:
            action_accuracy = (
                torch.sum(predicted_action == next_action).type(torch.float)
                / predicted_action.shape[0]
            )
        else:
            action_accuracy = 0

        token_prediction[token_labels == -1] = -1

        ignored_tokens_no = torch.sum(token_labels == -1)
        tokens_left = (
            token_prediction.shape[0] * token_prediction.shape[1]
        ) - ignored_tokens_no
        tokens_left = tokens_left.type(torch.float)

        token_accuracy = (
            torch.sum(token_prediction == token_labels).type(torch.float)
            - ignored_tokens_no
        ) / tokens_left

        return (
            loss,
            mask_loss,
            next_loss,
            token_loss,
            words_accuracy,
            action_accuracy,
            token_accuracy,
        )
