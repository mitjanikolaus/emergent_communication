import itertools
import math
import random
from collections import defaultdict

import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn, jit
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.nn import ModuleList, Parameter

import pandas as pd

from language_analysis import compute_topsim, compute_entropy, compute_posdis, compute_bosdis
from utils import MeanBaseline, find_lengths, NoBaseline


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def forward(self, input, state):
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, cy


class BaseEncoder(nn.Module):
    """encoder used for both the ModifSender and the ModifReceiver"""

    def __init__(self, input_size, hidden_size):
        super(BaseEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.sem_embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        batch_size = input.size(0)
        embedded = self.embedding(input)
        embedded = embedded.view(-1, batch_size, self.hidden_size)
        hidden = torch.zeros((1, batch_size, self.hidden_size)).type_as(embedded)
        output, hidden = self.gru(
            embedded, hidden
        )  # [seq_len, bs, hid_dim(*2 if bidirectional)]
        output = output.transpose(1, 0)  # [bs, seq_len, hid_dim(*2 if bidirectional)]
        sem_embs = self.sem_embedding(input)

        return output, hidden, sem_embs


class AttnMasked(nn.Module):
    """
    implementation taken from B.Lake's meta_seq2seq code:
    https://github.com/facebookresearch/meta_seq2seq/blob/59c3b4aafebf387bcd4e45626d8d91b66e6e5dff/model.py#L223
    """

    def __init__(self):
        super(AttnMasked, self).__init__()

    def forward(self, Q, K, V, key_length_mask):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #  key_length_mask: mask telling me which positions to ignore (True)
        #    and which to consider (False),
        #    the True/False assignment is given by the torch.masked_fill_,
        #    this method fills in a value for each True position;
        #    batch_size x

        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda:
            query_dim = query_dim.cuda()
        attn_weights = torch.bmm(
            Q, K.transpose(1, 2)
        )  # batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights.masked_fill_(key_length_mask, -1000)
        attn_weights = F.softmax(
            attn_weights, dim=2
        )  # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights, V)  # batch_size x n_queries x value_dim
        return R, attn_weights


class SenderDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, max_length):
        super(SenderDecoder, self).__init__()
        self.hid_dim = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size + 3, self.hid_dim)
        # +3 for [fake EOS emb, fake SOS emb, fake EOS semantic emb]
        self.gru = nn.GRU(self.hid_dim, self.hid_dim)
        self.attn = AttnMasked()
        self.out = nn.Linear(self.hid_dim, self.output_size)

    def get_per_step_logits(self, input, hidden, keys, values, length_mask):
        embedded = self.embedding(input)  # [1, bs, hid_dim]
        output, hidden = self.gru(
            embedded, hidden
        )  # [1, bs, hid_dim], [1, bs, hid_dim]
        # Attention
        queries = output.transpose(1, 0)  # [bs, 1, hid_dim]
        weighted_values, weights = self.attn(
            queries, keys, values, length_mask
        )  # [bs, 1, hid_dim], [bs, 1, max_len]

        logits = self.out(weighted_values[:, 0] + output[0])
        return logits, hidden, weights[:, 0, :]

    def init_batch(self, encoder_outputs, sem_embs):
        batch_size = encoder_outputs.size(0)
        input = (
            torch.zeros(1, batch_size, dtype=torch.long, device=encoder_outputs.device)
            + self.output_size
            + 2
        )  # [1, bs]
        hidden = torch.zeros(1, batch_size, self.hid_dim, device=encoder_outputs.device)  # [1, bs, hid_dim]

        fake_EOS = self.embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_outputs.device) + self.output_size
        )
        fake_EOS_sem = self.embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_outputs.device)
            + self.output_size
            + 1
        )
        keys = torch.cat([encoder_outputs, fake_EOS], dim=1)  # [bs, max_len, hid_dim]
        values = torch.cat([sem_embs, fake_EOS_sem], dim=1)  # [bs, max_len, hid_dim]
        length_mask = torch.zeros(
            batch_size, 1, keys.shape[1], dtype=torch.bool
        ).type_as(encoder_outputs)

        return batch_size, input, hidden, keys, values, length_mask

    def forward(self, encoder_outputs, sem_embs, lengths=None):
        bs, input, hidden, keys, values, length_mask = self.init_batch(
            encoder_outputs, sem_embs
        )
        sequence, per_step_logits, entropy, attn_weights = [], [], [], []
        for _ in range(self.max_length):
            logits, hidden, weights = self.get_per_step_logits(
                input, hidden, keys, values, length_mask
            )
            distr = Categorical(logits=logits)

            entropy.append(distr.entropy())
            if self.training:
                x = distr.sample()
            else:
                x = logits.argmax(dim=-1)
            per_step_logits.append(distr.log_prob(x))

            sequence.append(x)
            input = x[None]
            attn_weights.append(weights)
        zeros = torch.zeros((bs, 1)).type_as(encoder_outputs)

        sequence = torch.stack(sequence, 1)  # [bs, max_len, out_dim]
        sequence = torch.cat(
            [sequence, zeros.long()], dim=-1
        )  # [bs, max_len + 1, out_dim]
        per_step_logits = torch.cat(
            [torch.stack(per_step_logits, 1), zeros], dim=-1
        )  # [bs, max_len + 1, out_dim]
        entropy = torch.cat(
            [torch.stack(entropy, 1), zeros], dim=-1
        )  # [bs, max_len + 1, out_dim]
        attn_weights = torch.stack(attn_weights, 1)

        return sequence, per_step_logits, entropy, attn_weights


class ReceiverDecoder(SenderDecoder):
    def init_batch(self, encoder_outputs, sem_embs):
        batch_size = encoder_outputs.size(0)
        input = (
            torch.zeros(1, batch_size, dtype=torch.long, device=encoder_outputs.device)
            + self.output_size
            + 2
        )  # [1, bs]
        hidden = torch.zeros(1, batch_size, self.hid_dim).type_as(encoder_outputs)  # [1, bs, hid_dim]
        keys = encoder_outputs  # [bs, max_len, hid_dim]
        values = sem_embs  # [bs, max_len, hid_dim]
        length_mask = torch.zeros(
            batch_size, 1, keys.shape[1], dtype=torch.bool, device=encoder_outputs.device)

        return batch_size, input, hidden, keys, values, length_mask

    def forward(self, encoder_outputs, sem_embs, lengths=None):

        batch_size, input, hidden, keys, values, length_mask = self.init_batch(
            encoder_outputs, sem_embs
        )
        # we want to ignore the previous dummy length_mask
        # length_mask is TRUE for positions which are to be IGNORED
        length_mask = (
            torch.arange(encoder_outputs.size(1)).type_as(encoder_outputs)[None, :]
            >= lengths[:, None]
        )
        length_mask = length_mask[:, None, :]  # [bs, 1, max_message_length]

        per_step_logits, attn_weights = [], []
        for i in range(self.max_length):
            logits, hidden, weights = self.get_per_step_logits(
                input, hidden, keys, values, length_mask
            )
            per_step_logits.append(logits)
            attn_weights.append(weights)
            input = logits.argmax(dim=-1)
            input = input[None]

        per_step_logits = torch.stack(per_step_logits, 1)  # [bs, n_attrs, n_values]
        attn_weights = torch.stack(attn_weights, 1)

        top_logits_ = entropy_ = torch.zeros(batch_size).type_as(encoder_outputs)
        return per_step_logits, top_logits_, entropy_, attn_weights


class AltSender(nn.Module):
    """enc-dec architecture implementing the sender in the communication game"""

    def __init__(self, vocab_size, hidden_size, num_features, num_values, max_len):
        super(AltSender, self).__init__()
        self.encoder = BaseEncoder(num_values, hidden_size)
        self.decoder = SenderDecoder(
            vocab_size + 1, hidden_size, max_len
        )

        self.num_features = num_features
        self.num_values = num_values

    def forward_first_turn(self, x):
        # change the egg-style format (concatenation of one-hot encodings) to `ordinary`
        #   input format (vector of indices):
        batch_size = x.size(0)
        x = (
            x.view(batch_size * self.num_features, self.num_values)
            .nonzero()[:, 1]
            .view(batch_size, self.num_features)
        )

        enc_output, hidden, sem_embs = self.encoder(x)
        sequence, top_logits, entropy, attn_weights = self.decoder(
            enc_output, sem_embs
        )
        return sequence, top_logits, entropy


class AltReceiver(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_features, num_values):
        super(AltReceiver, self).__init__()
        self.encoder = BaseEncoder(vocab_size + 1, hidden_size)
        self.decoder = ReceiverDecoder(
            num_values, hidden_size, num_features
        )

        self.num_features = num_features
        self.num_values = num_values

    def forward_first_turn(self, message, message_lengths):
        enc_output, hidden, sem_embs = self.encoder(message)
        per_step_logits, logits, entropy, attn_weights = self.decoder(
            enc_output, sem_embs, message_lengths
        )
        per_step_logits = per_step_logits.view(-1, self.num_features * self.num_values)
        return per_step_logits, None, logits, entropy


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, max_len, n_features, n_values, layer_norm, num_layers
    ):
        super(Receiver, self).__init__()
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.embedding_perc = nn.Embedding(vocab_size+1, embed_dim)
        self.embedding_prod = nn.Embedding(vocab_size+1, embed_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len

        lstm_cell = LayerNormLSTMCell if layer_norm else nn.LSTMCell
        self.cells_perception = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )
        self.cells_production = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )

        self.fc1 = nn.Linear(n_features*n_values, hidden_size)

        self.linear_out = nn.Linear(hidden_size, n_features*n_values)

        self.embed_message = nn.Linear(hidden_size, hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.attn = nn.Linear(hidden_size*2, hidden_size*2)

    def forward_first_turn(self, messages, message_lengths):
        # Encode message
        batch_size = messages.shape[0]
        embedded_message = self.embedding_perc(messages)

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)]
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)
        ]

        max_message_len = embedded_message.shape[1]
        hidden_states = torch.zeros((batch_size, max_message_len, self.hidden_size)).type_as(embedded_message)
        for step in range(max_message_len):
            lstm_input = embedded_message[:, step]
            for i, layer in enumerate(self.cells_perception):
                h_t, c_t = layer(lstm_input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                lstm_input = h_t

            hidden_states[:, step] = h_t
        # TODO: verify message lengths
        encoded_message = hidden_states[range(batch_size), message_lengths-1]

        output = self.linear_out(encoded_message)

        # Create response:
        prev_hidden = [self.embed_message(encoded_message)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(encoded_message) for _ in range(self.num_layers)
        ]
        input = torch.stack([self.sos_embedding] * batch_size)

        sequence = []
        logits = []
        entropy = []
        all_step_logits = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells_production):
                h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))
            all_step_logits.append(step_logits)
            input = self.embedding_prod(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)
        all_step_logits = torch.stack(all_step_logits).permute(1, 0, 2)

        zeros = torch.zeros((sequence.size(0), 1)).type_as(encoded_message)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return output, sequence, logits, entropy, all_step_logits, encoded_message

    def forward_second_turn(self, encoded_messages_1, messages_2, message_lengths_2):
        batch_size = encoded_messages_1.shape[0]

        embedded_message = self.embedding_perc(messages_2)

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)]
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)
        ]

        max_message_len = embedded_message.shape[1]
        hidden_states = torch.zeros((batch_size, max_message_len, self.hidden_size)).type_as(embedded_message)
        for step in range(max_message_len):
            lstm_input = embedded_message[:, step]
            for i, layer in enumerate(self.cells_perception):
                h_t, c_t = layer(lstm_input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                lstm_input = h_t

            hidden_states[:, step] = h_t

        encoded_messages_2 = hidden_states[range(batch_size), message_lengths_2-1]

        encoded_messages = torch.cat((encoded_messages_1, encoded_messages_2), dim=-1)
        attn_weights = F.softmax(self.attn(encoded_messages).reshape(batch_size, self.hidden_size, 2), dim=-1)
        encoded_messages = torch.sum(attn_weights * encoded_messages.reshape(batch_size, self.hidden_size, -1), dim=-1)
        output = self.linear_out(encoded_messages)

        return output


class ReceiverMLP(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, n_features, n_values, max_message_len
    ):
        super(ReceiverMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim)
        self.linear_message = nn.Linear(embed_dim * (max_message_len + 1), embed_dim)

        self.linear_out = nn.Linear(embed_dim, n_features*n_values)

    def forward(self, batch):
        message, input, message_lengths = batch
        batch_size = message.shape[0]

        embedded_message = self.embedding(message)
        embedded_message = F.relu(embedded_message)
        embedded_message = self.linear_message(embedded_message.reshape(batch_size, -1))
        embedded_message = F.relu(embedded_message)

        output = self.linear_out(embedded_message)

        return output


class Sender(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_values,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        layer_norm,
        num_layers,
    ):
        super(Sender, self).__init__()
        self.max_len = max_len

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.embed_input = nn.Linear(n_features*n_values, hidden_size)

        self.linear_in_perc = nn.Linear(n_features * n_values, hidden_size)
        self.linear_in_prod = nn.Linear(hidden_size * 2, hidden_size)

        self.embed_input_lstm = nn.Linear(hidden_size, hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding_perc = nn.Embedding(vocab_size, embed_dim)
        self.embedding_prod = nn.Embedding(vocab_size, embed_dim)
        self.embedding_prod_turn_2 = nn.Embedding(vocab_size, embed_dim)


        self.linear_predict_noise_loc = nn.Linear(hidden_size, max_len)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        lstm_cell = LayerNormLSTMCell if layer_norm else nn.LSTMCell

        self.cells_perception = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

        #TODO: separate LSTMs for production turn 1/ turn 2?
        self.cells_production = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

        self.cells_production_turn_2 = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

    def forward_first_turn(self, input_objects):
        batch_size = input_objects.shape[0]

        prev_hidden = [self.linear_in_perc(input_objects)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(input_objects) for _ in range(self.num_layers)
        ]

        input = torch.stack([self.sos_embedding] * batch_size)

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells_production):
                h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                input_objects = distr.sample()
            else:
                input_objects = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(input_objects))

            input = self.embedding_prod(input_objects)
            sequence.append(input_objects)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).type_as(input_objects)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def forward_second_turn(self, input_objects, messages, message_lengths):
        # Encode message
        batch_size = messages.shape[0]
        embedded_message = self.embedding_perc(messages)

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in
                    range(self.num_layers)]
        prev_c = [
         torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)
        ]

        max_message_len = embedded_message.shape[1]
        hidden_states = torch.zeros((batch_size, max_message_len, self.hidden_size)).type_as(embedded_message)
        for step in range(max_message_len):
            lstm_input = embedded_message[:, step]
            for i, layer in enumerate(self.cells_perception):
                h_t, c_t = layer(lstm_input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                lstm_input = h_t

        hidden_states[:, step] = h_t

        encoded_message = hidden_states[range(batch_size), message_lengths-1]

        input_msg = self.embed_input_lstm(encoded_message)

        input_obj = self.embed_input(input_objects)

        prev_hidden = torch.cat((input_obj, input_msg), dim=1)

        prev_hidden = [self.linear_in_prod(prev_hidden)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        input = torch.stack([self.sos_embedding] * batch_size)

        # prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(input) for _ in range(self.num_layers)]
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(input) for _ in range(self.num_layers)
        ]

        sequence = []
        logits = []
        entropy = []
        all_step_logits = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells_production_turn_2):
                h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                messages = distr.sample()
            else:
                messages = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(messages))
            all_step_logits.append(step_logits)

            input = self.embedding_prod_turn_2(messages)
            sequence.append(messages)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)
        all_step_logits = torch.stack(all_step_logits).permute(1, 0, 2)

        zeros = torch.zeros((sequence.size(0), 1)).type_as(messages)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        output_noise_location = self.linear_predict_noise_loc(encoded_message)

        return sequence, logits, entropy, output_noise_location, all_step_logits


class OptimalSender(pl.LightningModule):
    # TODO: update for multiple turns
    def __init__(
        self,
        n_features,
        n_values,
        vocab_size,
        max_len,
    ):
        super(OptimalSender, self).__init__()
        self.max_len = max_len

        self.n_features = n_features
        self.n_values = n_values

        self.vocab_size = vocab_size

        assert n_values + 2 <= vocab_size   # +1 for case if value is not set and + 1 for EOS token
        assert n_features + 1 <= max_len    # + 1 to encode speech act

    def one_hot_to_message(self, intent_objects):
        values = []
        for i in range(self.n_features):
            # Cut out relevant range for this feature
            relevant_range = intent_objects[:, i * self.n_values:(i + 1) * self.n_values]
            # Prepend zeros for case if feature is not set
            zeros = torch.zeros(relevant_range.shape[0]).unsqueeze(1).type_as(intent_objects)
            relevant_range = torch.cat((zeros, relevant_range), dim=1)

            value = torch.argmax(relevant_range, dim=1)
            values.append(value)

        return torch.stack(values).T

    def create_messages(self, intents):
        message_contents = self.one_hot_to_message(objects)

        messages = torch.cat((speech_act_codes, message_contents), dim=1)

        # Add 1 to avoid zeros, these are reserved as EOS token
        messages = messages + 1
        return messages

    def forward(self, x):
        batch_size = x.shape[0]

        messages = self.create_messages(x)

        zeros = torch.zeros((batch_size, 1)).type_as(x)

        sequences = torch.cat([messages.long(), zeros.long()], dim=1)
        logits = torch.zeros_like(sequences)
        entropy = torch.zeros_like(sequences)

        return sequences, logits, entropy


class SenderReceiver(pl.LightningModule):
    # TODO: update for multiple turns
    def __init__(
        self,
        speech_acts,
        n_features,
        n_values,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        sender_layer_norm,
        receiver_layer_norm,
        num_layers=1,
    ):
        super(SenderReceiver, self).__init__()
        self.max_len = max_len
        self.speech_acts = speech_acts
        self.embed_input = nn.Linear(n_features*n_values, embed_dim)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        if sender_layer_norm != receiver_layer_norm:
            raise ValueError("Joint Sender and Receiver requires both sender_layer_norm and receiver_layer_norm to be "
                             "set to true or false at the same time")
        lstm_cell = LayerNormLSTMCell if sender_layer_norm else nn.LSTMCell
        self.cells = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )


    def forward(self, batch):
        if isinstance(batch, tuple):
            return self.forward_receiver(batch)
        else:
            return self.forward_sender(batch)

    def forward_sender(self, x):
        batch_size = x.shape[0]

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(x) for _ in range(self.num_layers)]

        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(x) for _ in range(self.num_layers)
        ]
        input = self.embed_input(x)

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).type_as(x)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def forward_receiver(self, batch):
        message, input_receiver, message_lengths = batch
        batch_size = message.shape[0]

        perplexities = []
        for objects_idx in range(input_receiver.shape[1]):
            input = self.embed_input(input_receiver[:,objects_idx])

            prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(input_receiver) for _ in
                           range(self.num_layers)]

            prev_c = [
                torch.zeros((batch_size, self.hidden_size)).type_as(input_receiver) for _ in range(self.num_layers)
            ]


            perplexities_objects = []
            for step in range(self.max_len):
                for i, layer in enumerate(self.cells):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                    prev_hidden[i] = h_t
                    input = h_t

                step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
                # TODO: EOS handling!
                ppls = step_logits[range(batch_size), message[:,step]]
                perplexities_objects.append(ppls)

                x = message[:, step]
                input = self.embedding(x)

            perplexities_objects = torch.stack(perplexities_objects).sum(dim=0)
            perplexities.append(perplexities_objects)

        perplexities = torch.stack(perplexities).permute(1,0)

        softmaxed = F.softmax(perplexities, dim=1)

        return softmaxed


class SignalingGameModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_hparams = AttributeDict(self.hparams["model"])

        self.init_agents()

        self.num_features = self.model_hparams.num_features
        self.num_values = self.model_hparams.num_values

        self.sender_entropy_coeff = self.model_hparams.sender_entropy_coeff
        self.receiver_entropy_coeff = self.model_hparams.receiver_entropy_coeff

        self.length_cost = self.model_hparams.length_cost

        if self.model_hparams.baseline_type == "mean":
            self.baselines = defaultdict(MeanBaseline)
        elif self.model_hparams.baseline_type == "none":
            self.baselines = defaultdict(NoBaseline)
        else:
            raise ValueError("Unknown baseline type: ", self.model_hparams.baseline_type)

        if not 0 < self.model_hparams.sender_learning_speed <= 1:
            raise ValueError("Sender learning speed should be between 0 and 1 ", self.model_hparams.sender_learning_speed)

        if not 0 < self.model_hparams.receiver_learning_speed <= 1:
            raise ValueError("Receiver learning speed should be between 0 and 1 ", self.model_hparams.receiver_learning_speed)

        self.token_noise = self.model_hparams["vocab_size"]
        self.automatic_optimization = False

    def init_agents(self):
        if self.model_hparams.symmetric:
            if self.model_hparams.num_senders != self.model_hparams.num_receivers:
                raise ValueError("Symmetric game requires same number of senders and receivers.")
            self.senders = ModuleList(
                [
                    SenderReceiver(self.model_hparams.num_features, self.model_hparams.num_values,
                        self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim,
                        self.model_hparams.sender_hidden_dim, self.model_hparams.max_len,
                        self.model_hparams.sender_layer_norm, self.model_hparams.receiver_layer_norm,
                        self.model_hparams.sender_num_layers)
                    for _ in range(self.model_hparams.num_senders * 2)
                ]
            )
            self.receivers = self.senders
        else:
            if self.model_hparams.optimal_sender:
                self.senders = ModuleList(
                    [
                        OptimalSender(
                               self.model_hparams.num_features, self.model_hparams.num_values,
                               self.model_hparams.vocab_size, self.model_hparams.max_len)
                        for _ in range(self.model_hparams.num_senders)
                    ]
                )
            else:
                if self.model_hparams.alt_agents:
                    self.senders = ModuleList(
                        [
                            AltSender(
                                self.model_hparams.vocab_size, self.model_hparams.sender_hidden_dim,
                                self.model_hparams.num_features, self.model_hparams.num_values,
                                self.model_hparams.max_len
                                )
                            for _ in range(self.model_hparams.num_senders)
                        ]
                    )
                else:
                    self.senders = ModuleList(
                        [
                            Sender(
                                   self.model_hparams.num_features, self.model_hparams.num_values,
                                   self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim,
                                   self.model_hparams.sender_hidden_dim, self.model_hparams.max_len,
                                   self.model_hparams.sender_layer_norm, self.model_hparams.sender_num_layers)
                            for _ in range(self.model_hparams.num_senders)
                        ]
                    )
            if self.model_hparams.alt_agents:
                self.receivers = ModuleList(
                    [
                        AltReceiver(self.model_hparams.vocab_size, self.model_hparams.receiver_hidden_dim,
                                    self.model_hparams.num_features, self.model_hparams.num_values)
                        for _ in range(self.model_hparams.num_receivers)
                    ]
                )
            else:
                self.receivers = ModuleList(
                    [
                        Receiver(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim,
                                 self.model_hparams.receiver_hidden_dim, self.model_hparams.max_len,
                                 self.model_hparams.num_features, self.model_hparams.num_values,
                                 self.model_hparams.receiver_layer_norm, self.model_hparams.receiver_num_layers)
                        for _ in range(self.model_hparams.num_receivers)
                    ]
                )

    def init_MLP_receivers(self):
        self.receivers = ModuleList(
            [
                ReceiverMLP(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim,
                         self.model_hparams.num_features, self.model_hparams.num_values,
                         self.model_hparams.max_len)
                for _ in range(self.model_hparams.num_receivers)
            ]
        )

    def freeze_senders(self):
        for sender in self.senders:
            for param in sender.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        if self.model_hparams.optimal_sender:
            optimizers_sender = []
        else:
            optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=1e-3) for sender in self.senders]
        optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=1e-3) for receiver in self.receivers]

        return tuple(itertools.chain(optimizers_sender, optimizers_receiver))

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        if self.model_hparams.symmetric:
            num_agents = self.model_hparams.num_senders + self.model_hparams.num_receivers
            sender_idx = random.choice(range(num_agents))
            receiver_idx = random.choice(range(num_agents))
            # Avoid communication within same agent
            while (sender_idx == receiver_idx):
                sender_idx = random.choice(range(num_agents))
                receiver_idx = random.choice(range(num_agents))

            opt_sender = optimizers[sender_idx]
            opt_receiver = optimizers[receiver_idx]

        else:
            # Sample sender and receiver for this batch:
            sender_idx = random.choice(range(self.model_hparams.num_senders))
            receiver_idx = random.choice(range(self.model_hparams.num_receivers))

            if self.model_hparams.optimal_sender:
                opt_sender = None
                opts_receiver = optimizers
                if self.model_hparams.num_receivers == 1:
                    opt_receiver = opts_receiver
                else:
                    opt_receiver = opts_receiver[receiver_idx]

            else:
                opts_sender = optimizers[:self.model_hparams.num_senders]
                opts_receiver = optimizers[self.model_hparams.num_senders:]

                opt_sender = opts_sender[sender_idx]
                opt_receiver = opts_receiver[receiver_idx]

        if opt_sender:
            opt_sender.zero_grad()
        opt_receiver.zero_grad()
        loss, acc = self.forward(batch, sender_idx, receiver_idx)
        self.manual_backward(loss)

        perform_sender_update = torch.rand(1) < self.model_hparams.sender_learning_speed
        if perform_sender_update and opt_sender:
            opt_sender.step()

        perform_receiver_update = torch.rand(1) < self.model_hparams.receiver_learning_speed
        if perform_receiver_update:
            opt_receiver.step()

        self.log(f"train_acc", acc.float().mean(), prog_bar=True, add_dataloader_idx=False)

    def forward(
        self, sender_input, sender_idx, receiver_idx, return_messages=False, disable_noise=False
    ):
        sender = self.senders[sender_idx]
        receiver = self.receivers[receiver_idx]
        batch_size = sender_input.shape[0]

        messages_sender_1, log_prob_s, entropy_s = sender.forward_first_turn(sender_input)
        messages_sender_1_lengths = find_lengths(messages_sender_1)
        self.log(f"message_lengths", messages_sender_1_lengths.float().mean() - 1)

        original_messages_sender_1 = messages_sender_1.detach().clone()
        if not disable_noise:
            messages_sender_1 = self.add_noise(messages_sender_1)

        receiver_output_1, messages_receiver_1, log_prob_r, entropy_r, all_step_logits_r, receiver_encoded_messages_1 = receiver.forward_first_turn(
            messages_sender_1, messages_sender_1_lengths
        )
        receiver_output_1 = receiver_output_1.view(batch_size, self.num_features, self.num_values)
        receiver_out_1_entropy = Categorical(logits=receiver_output_1).entropy()
        self.log(f"receiver_out_1_entropy", receiver_out_1_entropy.float().mean())

        if self.model_hparams.multi_turn:
            messages_receiver_1_lengths = find_lengths(messages_receiver_1)
            self.log(f"message_lengths_receiver", messages_receiver_1_lengths.float().mean() - 1)

            messages_sender_2, log_prob_s_2, entropy_s_2, out_noise_loc, all_step_logits_s_2 = sender.forward_second_turn(sender_input, messages_receiver_1, messages_receiver_1_lengths)

            messages_sender_2_lengths = find_lengths(messages_sender_2)
            self.log(f"message_lengths_sender_second_turn", messages_sender_2_lengths.float().mean() - 1)

            if not disable_noise:
                messages_sender_2 = self.add_noise(messages_sender_2)

            receiver_output_2 = receiver.forward_second_turn(receiver_encoded_messages_1, messages_sender_2, messages_sender_2_lengths)
            receiver_output_2 = receiver_output_2.view(batch_size, self.num_features, self.num_values)
            receiver_out_2_entropy = Categorical(logits=receiver_output_2).entropy()
            self.log(f"receiver_out_2_entropy", receiver_out_2_entropy.float().mean())

        sender_input = sender_input.view(batch_size, self.num_features, self.num_values)
        acc_first_turn = (torch.sum((receiver_output_1.argmax(dim=-1) == sender_input.argmax(dim=-1)
                          ).detach(), dim=1) == self.num_features).float()
        self.log(f"acc_first_turn", acc_first_turn.mean())

        if self.model_hparams.multi_turn:
            acc = (torch.sum((receiver_output_2.argmax(dim=-1) == sender_input.argmax(dim=-1)
                          ).detach(), dim=1) == self.num_features).float()
        else:
            acc = acc_first_turn

        receiver_output_1 = receiver_output_1.view(batch_size * self.num_features, self.num_values)

        labels = sender_input.argmax(dim=-1).view(batch_size * self.num_features)
        receiver_loss_1 = F.cross_entropy(receiver_output_1, labels, reduction="none").view(batch_size, self.num_features).mean(dim=-1)
        receiver_loss = receiver_loss_1
        if self.model_hparams.multi_turn:
            receiver_output_2 = receiver_output_2.view(batch_size * self.num_features, self.num_values)
            receiver_loss_2 = F.cross_entropy(receiver_output_2, labels, reduction="none").view(batch_size, self.num_features).mean(dim=-1)
            receiver_loss = receiver_loss_2

        self.log(f"receiver_loss", receiver_loss.mean())

        assert len(receiver_loss) == batch_size

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros(batch_size).type_as(sender_input)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros(batch_size).type_as(sender_input)

        for i in range(messages_sender_1.size(1)):
            not_eosed = (i < messages_sender_1_lengths).float()
            effective_entropy_s_1 += entropy_s[:, i] * not_eosed
            effective_log_prob_s_1 += log_prob_s[:, i] * not_eosed
        effective_entropy_s_1 = effective_entropy_s_1 / messages_sender_1_lengths.float()

        entropy_loss = effective_entropy_s_1.mean() * self.sender_entropy_coeff
        log_prob_s = effective_log_prob_s_1

        baseline = self.baselines["length_sender_1"].predict(messages_sender_1_lengths.device)
        policy_length_loss = (messages_sender_1_lengths.float() - baseline) * self.length_cost * effective_log_prob_s_1

        if self.model_hparams.multi_turn:
            effective_entropy_s_2 = torch.zeros(batch_size).type_as(sender_input)
            effective_log_prob_s_2 = torch.zeros(batch_size).type_as(sender_input)
            for i in range(messages_sender_2.size(1)):
                not_eosed = (i < messages_sender_2_lengths).float()
                effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed
                effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed
            effective_entropy_s_2 = effective_entropy_s_2 / messages_sender_2_lengths.float()

            effective_entropy_r = torch.zeros(batch_size).type_as(sender_input)
            effective_log_prob_r = torch.zeros(batch_size).type_as(sender_input)
            for i in range(messages_receiver_1.size(1)):
                not_eosed = (i < messages_receiver_1_lengths).float()
                effective_entropy_r += entropy_r[:, i] * not_eosed
                effective_log_prob_r += log_prob_r[:, i] * not_eosed
            effective_entropy_r = effective_entropy_r / messages_receiver_1_lengths.float()

            entropy_loss = (effective_entropy_s_1.mean() * self.sender_entropy_coeff
                                + effective_entropy_s_2.mean() * self.sender_entropy_coeff
                                + effective_entropy_r.mean() * self.receiver_entropy_coeff)
            log_prob_s = effective_log_prob_s_1 + effective_log_prob_s_2

            baseline = self.baselines["length_receiver_1"].predict(messages_receiver_1_lengths.device)
            policy_length_loss += (messages_receiver_1_lengths.float() - baseline) * self.length_cost * effective_log_prob_r

            baseline = self.baselines["length_sender_2"].predict(messages_sender_2_lengths.device)
            policy_length_loss += (messages_sender_2_lengths.float() - baseline) * self.length_cost * effective_log_prob_s_2

        self.log(f"entropy_loss", entropy_loss.mean())

        loss_baseline = self.baselines["loss"].predict(receiver_loss.device)
        policy_loss = (
            (receiver_loss.detach() - loss_baseline) * log_prob_s
        ).mean()

        policy_length_loss = policy_length_loss.mean()

        self.log(f"policy_loss", policy_loss.mean())
        self.log(f"policy_length_loss", policy_length_loss.mean())

        optimized_loss = policy_length_loss + policy_loss - entropy_loss

        # add the differentiable loss terms
        optimized_loss += receiver_loss.mean()

        if self.model_hparams.multi_turn:
            if self.model_hparams.receiver_aux_loss:
                self.log(f"receiver_aux_loss", receiver_loss_1.mean())
                optimized_loss += receiver_loss_1.mean()

            if self.model_hparams.receiver_aux_loss_2 and not disable_noise:
                if self.model_hparams.noise == "one_symbol":
                    noise_locations = messages_sender_1.argmax(dim=1)
                    # Encourage receiver to send noise location in first token
                    receiver_noise_loss = F.nll_loss(all_step_logits_r[:, 0], noise_locations, reduction="none")
                    self.log(f"receiver_aux_loss_2", receiver_noise_loss.mean())

                    optimized_loss += receiver_noise_loss.mean()
                else:
                    raise NotImplementedError()

            if self.model_hparams.receiver_aux_loss_3 and not disable_noise:
                predicted_noise_locations = receiver_out_2_entropy.mean(dim=1) > receiver_out_2_entropy.mean()
                # Loss: push length to be greater 0 only if noise is present
                receiver_aux_loss_3 = (predicted_noise_locations != (messages_receiver_1_lengths.float() - 1 > 0)).float()
                self.log(f"receiver_aux_loss_3", receiver_aux_loss_3.mean())
                receiver_aux_loss_3 *= effective_log_prob_r

                optimized_loss += receiver_aux_loss_3.mean()

            if self.model_hparams.sender_aux_loss and not disable_noise:
                labels_noise_loc = torch.nonzero(messages_sender_1 == self.token_noise)[:, 1]
                sender_aux_loss = F.cross_entropy(out_noise_loc, labels_noise_loc, reduction="none")
                self.log(f"sender_aux_loss", sender_aux_loss.mean())
                optimized_loss += sender_aux_loss.mean()

            if self.model_hparams.sender_aux_loss_2:
                if self.model_hparams.noise == "one_symbol":
                    noise_locations = messages_sender_1.argmax(dim=1)
                    content_noise_locations = original_messages_sender_1[range(batch_size), noise_locations]
                    # Encourage sender to send content from noise location again
                    sender_noise_loss = F.nll_loss(all_step_logits_s_2[:, 0], content_noise_locations, reduction="none")
                    self.log(f"sender_aux_loss_2", sender_noise_loss.mean())
                    optimized_loss += sender_noise_loss.mean()
                else:
                    raise NotImplementedError()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length_sender_1"].update(messages_sender_1_lengths.float())
            if self.model_hparams.multi_turn:
                self.baselines["length_receiver_1"].update(messages_receiver_1_lengths.float())
                self.baselines["length_sender_2"].update(messages_sender_2_lengths.float())

        if return_messages:
            return optimized_loss, acc, messages_sender_1
        else:
            return optimized_loss, acc

    def add_noise(self, messages):
        if self.model_hparams.noise == "one_symbol":
            #TODO: do not overwrite zeroes?
            indices = torch.randint(0, messages.size(1)-1, (messages.size(0),))
            messages[range(messages.size(0)), indices] = self.token_noise
        else:
            if self.model_hparams.noise > 0:
                indices = torch.multinomial(torch.tensor([1 - self.model_hparams.noise, self.model_hparams.noise]), messages.shape[0] * messages.shape[1], replacement=True)
                indices = indices.reshape(messages.shape[0], messages.shape[1])
                # Replace all randomly selected values (but only if they are not EOS symbols (0))
                messages[(indices == 1) & (messages.to(indices.device) != 0)] = self.token_noise
        return messages

    def on_validation_epoch_start(self):
        # Sample agent indices for this validation epoch
        if self.model_hparams.symmetric:
            num_agents = self.model_hparams.num_senders + self.model_hparams.num_receivers
            self.val_epoch_sender_idx = random.choice(range(num_agents))
            self.val_epoch_receiver_idx = random.choice(range(num_agents))
            # Avoid communication within same agent
            while (self.val_epoch_sender_idx == self.val_epoch_receiver_idx):
                self.val_epoch_sender_idx = random.choice(range(num_agents))
                self.val_epoch_receiver_idx = random.choice(range(num_agents))
        else:
            self.val_epoch_sender_idx = random.choice(range(self.model_hparams.num_senders))
            self.val_epoch_receiver_idx = random.choice(range(self.model_hparams.num_receivers))
        print(f"\nValidating for sender {self.val_epoch_sender_idx} and receiver {self.val_epoch_receiver_idx}:\n")

    def validation_step(self, sender_input, batch_idx, dataloader_idx):
        sender_idx = self.val_epoch_sender_idx
        receiver_idx = self.val_epoch_receiver_idx
        if dataloader_idx == 0:
            # Generalization
            _, acc = self.forward(sender_input, sender_idx, receiver_idx)
            _, acc_no_noise = self.forward(sender_input, sender_idx, receiver_idx, disable_noise=True)

            return acc, acc_no_noise

        elif dataloader_idx == 1:
            # Language analysis (on train set data)
            _, acc_no_noise, messages = self.forward(sender_input, sender_idx, receiver_idx, return_messages=True, disable_noise=True)

            return sender_input, messages, acc_no_noise

    def validation_epoch_end(self, validation_step_outputs):
        # Generalization:
        accs = torch.cat([acc for acc, _ in validation_step_outputs[0]])
        accs_no_noise = torch.cat([acc_no_noise for _, acc_no_noise in validation_step_outputs[0]])
        test_acc = accs.mean().item()
        test_acc_no_noise = accs_no_noise.mean().item()
        self.log("test_acc", test_acc, prog_bar=True, add_dataloader_idx=False)
        self.log("test_acc_no_noise", test_acc_no_noise, prog_bar=True, add_dataloader_idx=False)
        print("test_acc: ", test_acc)
        print("test_acc_no_noise: ", test_acc_no_noise)

        # Language analysis (on train set data)
        language_analysis_results = validation_step_outputs[1]
        train_acc_no_noise = torch.cat([acc for _, _, acc in language_analysis_results])
        self.log("train_acc_no_noise", train_acc_no_noise, prog_bar=True, add_dataloader_idx=False)

        meanings = torch.cat([meaning.cpu() for meaning, _, _ in language_analysis_results])
        messages = torch.cat([message.cpu() for _, message, _ in language_analysis_results])
        self.analyze_language(messages, meanings)

    def analyze_language(self, messages, meanings):
        # Remove trailing 0s
        assert torch.all(messages[:, -1] == 0), "Messages do not contain trailing 0"
        messages = messages[:, :-1]

        num_unique_messages = len(messages.unique(dim=0))
        self.log("num_unique_messages", float(num_unique_messages))

        meanings_strings = pd.DataFrame(meanings).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)

        num_digits = int(math.log10(self.model_hparams.vocab_size))
        messages_strings = pd.DataFrame(messages).apply(lambda row: "".join([s.zfill(num_digits) for s in row.astype(int).astype(str)]), axis=1)
        messages_df = pd.DataFrame([meanings_strings, messages_strings]).T
        messages_df.rename(columns={0: 'meaning', 1: 'message'}, inplace=True)
        messages_df.to_csv(f"{self.logger.log_dir}/messages.csv", index=False)

        if self.model_hparams.log_entropy_on_validation:
            entropy = compute_entropy(messages.numpy())
            self.log("message_entropy", entropy, prog_bar=True)
            print("message_entropy: ", entropy)

        if self.model_hparams.log_topsim_on_validation:
            topsim = compute_topsim(meanings, messages)
            self.log("topsim", topsim, prog_bar=True)
            print("Topsim: ", topsim)

        if self.model_hparams.log_posdis_on_validation:
            posdis = compute_posdis(self.num_features, self.num_values, meanings, messages)
            self.log("posdis", posdis, prog_bar=True)
            print("posdis: ", posdis)

        if self.model_hparams.log_bosdis_on_validation:
            bosdis = compute_bosdis(meanings, messages, self.model_hparams["vocab_size"])
            self.log("bosdis", bosdis, prog_bar=True)
            print("bodis: ", bosdis)


    def on_fit_start(self):
        # Set which metrics to use for hyperparameter tuning
        metrics = ["test_acc"]
        metrics.append("topsim")
        self.logger.log_hyperparams(self.hparams, {m: 0 for m in metrics})
