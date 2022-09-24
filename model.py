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
from torch.nn import ModuleList, Parameter, GRUCell

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


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        i2h = self.i2h(x)
        h2h = self.h2h(h)

        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))

        h_t = torch.mul(1-z_t, h ) + torch.mul(z_t, h_hat)

        h_t = h_t.view(h_t.size(0), -1)
        return h_t


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, max_len, n_attributes, n_values, layer_norm, num_layers, open_cr
    ):
        super(Receiver, self).__init__()

        # Add one symbol for noise treatment
        vocab_size_perception = vocab_size + 1

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.linear_in_perc = nn.Linear(hidden_size, hidden_size)

        self.embedding_perc = nn.Embedding(vocab_size_perception, embed_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len

        if layer_norm:
            rnn_cell = LayerNormGRUCell
        else:
            rnn_cell = GRUCell

        self.feedback_cells = nn.ModuleList(
            [
                rnn_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )

        self.out_cells = nn.ModuleList(
            [
                rnn_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )

        if open_cr:
            # Open clarification request: receiver can only answer with binary feedback signal
            self.hidden_to_output = nn.Linear(hidden_size, 2)
        else:
            self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.linear_out = nn.Linear(hidden_size, n_attributes * n_values)

        self.attn = nn.Linear(hidden_size, max_len * n_attributes * n_values)

    def forward_first_turn(self, messages):
        batch_size = messages.shape[0]

        prev_hidden = [torch.zeros((batch_size, self.hidden_size), dtype=torch.float, device=messages.device) for _ in
                       range(self.num_layers)]

        return self.forward(messages, prev_hidden)

    def forward(self, messages, prev_hidden):
        rnn_input = self.embedding_perc(messages)

        for i, layer in enumerate(self.feedback_cells):
            h_t = layer(rnn_input, prev_hidden[i])
            prev_hidden[i] = h_t
            rnn_input = h_t

        step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
        distr = Categorical(logits=step_logits)
        entropy = distr.entropy()

        if self.training:
            output_token = distr.sample()
        else:
            output_token = step_logits.argmax(dim=1)

        logits = distr.log_prob(output_token)

        return output_token, entropy, logits, prev_hidden

    def forward_output(self, messages_sender, message_lengths):
        batch_size = messages_sender.shape[0]

        embedded = self.embedding_perc(messages_sender)

        prev_hidden = [torch.zeros((batch_size, self.hidden_size), dtype=torch.float, device=messages_sender.device) for _ in
                       range(self.num_layers)]

        max_message_length = max(message_lengths)
        hidden_states = torch.zeros((batch_size, max_message_length, self.hidden_size))

        for step in range(max_message_length):
            rnn_input = embedded[:, step]
            for i, layer in enumerate(self.out_cells):
                h_t = layer(rnn_input, prev_hidden[i])
                prev_hidden[i] = h_t
                rnn_input = h_t

            hidden_states[:, step] = h_t

        last_hidden_states = hidden_states[range(batch_size), message_lengths-1]

        out = self.linear_out(last_hidden_states)

        return out


class ReceiverMLP(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, n_attributes, n_values, max_message_len
    ):
        super(ReceiverMLP, self).__init__()

        # Add one symbol for noise treatment
        vocab_size += 1

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear_message = nn.Linear(embed_dim * (max_message_len + 1), embed_dim)

        self.linear_out = nn.Linear(embed_dim, n_attributes*n_values)

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
        n_attributes,
        n_values,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        layer_norm,
        num_layers,
        feedback,
        open_cr = False,
    ):
        super(Sender, self).__init__()

        self.max_len = max_len

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.embed_input = nn.Linear(n_attributes*n_values, hidden_size)

        self.linear_in_perc = nn.Linear(n_attributes * n_values, hidden_size)
        self.linear_in_prod = nn.Linear(hidden_size * 2, hidden_size)

        self.feedback = feedback
        if self.feedback:
            self.linear_in_response = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.linear_in_response = nn.Linear(embed_dim, embed_dim)

        self.embed_input_rnn = nn.Linear(hidden_size, hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding_perc = nn.Embedding(vocab_size, embed_dim)

        # Vocab size + 1 for noise handling
        vocab_size_noise = vocab_size + 1
        self.embedding_prod = nn.Embedding(vocab_size_noise, embed_dim)

        if open_cr:
            self.embedding_response = nn.Embedding(2, embed_dim)
        else:
            self.embedding_response = nn.Embedding(vocab_size, embed_dim)

        self.linear_predict_noise_loc = nn.Linear(hidden_size, max_len)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.feedback = feedback

        if layer_norm:
            rnn_cell = LayerNormGRUCell
        else:
            rnn_cell = GRUCell

        self.cells = nn.ModuleList(
            [
                rnn_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

    def forward_first_turn(self, input_objects):
        batch_size = input_objects.shape[0]

        prev_hidden = [self.linear_in_perc(input_objects)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        input = torch.stack([self.sos_embedding] * batch_size)

        return self.forward(input, prev_hidden)

    def forward_subsequent_turn(self, output_last_turn, prev_hidden, input_response):
        embedded_output_last_turn = self.embedding_prod(output_last_turn)

        if self.feedback:
            embedded_input = self.embedding_response(input_response)
            inputs = torch.cat((embedded_input, embedded_output_last_turn), dim=1)
        else:
            inputs = embedded_output_last_turn

        input = self.linear_in_response(inputs)

        return self.forward(input, prev_hidden)

    def forward(self, input, prev_hidden):
        for i, layer in enumerate(self.cells):
            h_t = layer(input, prev_hidden[i])
            prev_hidden[i] = h_t
            input = h_t

        step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
        distr = Categorical(logits=step_logits)

        if self.training:
            output_tokens = distr.sample()
        else:
            output_tokens = step_logits.argmax(dim=1)

        entropy = distr.entropy()
        logits = distr.log_prob(output_tokens)

        return output_tokens, entropy, logits, prev_hidden


class OptimalSender(pl.LightningModule):
    # TODO: update for multiple turns
    def __init__(
        self,
        n_attributes,
        n_values,
        vocab_size,
        max_len,
    ):
        super(OptimalSender, self).__init__()

        self.max_len = max_len

        self.n_attributes = n_attributes
        self.n_values = n_values

        self.vocab_size = vocab_size

        assert n_values + 2 <= vocab_size   # +1 for case if value is not set and + 1 for EOS token
        assert n_attributes + 1 <= max_len    # + 1 to encode speech act

    def one_hot_to_message(self, intent_objects):
        values = []
        for i in range(self.n_attributes):
            # Cut out relevant range for this feature
            relevant_range = intent_objects[:, i * self.n_values:(i + 1) * self.n_values]
            # Prepend zeros for case if feature is not set
            zeros = torch.zeros(relevant_range.shape[0]).unsqueeze(1).type_as(intent_objects)
            relevant_range = torch.cat((zeros, relevant_range), dim=1)

            value = torch.argmax(relevant_range, dim=1)
            values.append(value)

        return torch.stack(values).T

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
        n_attributes,
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

        # Add one symbol for noise treatment
        vocab_size_perception = vocab_size + 1

        self.max_len = max_len
        self.speech_acts = speech_acts
        self.embed_input = nn.Linear(n_attributes*n_values, embed_dim)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size_perception, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        if sender_layer_norm != receiver_layer_norm:
            raise ValueError("Joint Sender and Receiver requires both sender_layer_norm and receiver_layer_norm to be "
                             "set to true or false at the same time")
        rnn_cell = GRUCell
        self.cells = nn.ModuleList(
            [
                rnn_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
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

        input = self.embed_input(x)

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                h_t = layer(input, prev_hidden[i])
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

            perplexities_objects = []
            for step in range(self.max_len):
                for i, layer in enumerate(self.cells):
                    h_t = layer(input, prev_hidden[i])
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
    def __init__(self, symmetric=False, optimal_sender=False, load_checkpoint=None, baseline_type="mean",
                 max_len=4, length_cost=0, num_attributes=4, num_values=4, num_senders=1, num_receivers=1,
                 receiver_embed_dim=30, receiver_num_layers=500, receiver_hidden_dim=500,
                 receiver_learning_speed=1, sender_embed_dim=5, sender_entropy_coeff=0.5,
                 receiver_entropy_coeff=0.5, sender_num_layers=1, receiver_layer_norm=False,
                 sender_layer_norm=False, sender_hidden_dim=500, sender_learning_speed=1,
                 vocab_size=5, noise=0, feedback=False, open_cr=False,
                 log_topsim_on_validation=False, log_posdis_on_validation=False,
                 log_bosdis_on_validation=False, log_entropy_on_validation=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.params = AttributeDict(self.hparams)

        self.init_agents()

        self.num_attributes = self.params.num_attributes
        self.num_values = self.params.num_values

        self.sender_entropy_coeff = self.params.sender_entropy_coeff
        self.receiver_entropy_coeff = self.params.receiver_entropy_coeff

        self.length_cost = self.params.length_cost

        if self.params.baseline_type == "mean":
            self.baselines = defaultdict(MeanBaseline)
        elif self.params.baseline_type == "none":
            self.baselines = defaultdict(NoBaseline)
        else:
            raise ValueError("Unknown baseline type: ", self.params.baseline_type)

        if not 0 < self.params.sender_learning_speed <= 1:
            raise ValueError("Sender learning speed should be between 0 and 1 ", self.params.sender_learning_speed)

        if not 0 < self.params.receiver_learning_speed <= 1:
            raise ValueError("Receiver learning speed should be between 0 and 1 ", self.params.receiver_learning_speed)

        self.token_noise = self.params["vocab_size"]
        self.automatic_optimization = False

        self.best_val_acc_no_noise = 0.0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--symmetric", default=False, action="store_true")
        parser.add_argument("--optimal-sender", default=False, action="store_true")
        parser.add_argument("--load-checkpoint", type=str, default=None)
        parser.add_argument("--baseline-type", type=str, default="mean")

        parser.add_argument("--num-attributes", type=int, default=4)
        parser.add_argument("--num-values", type=int, default=4)
        parser.add_argument("--num-senders", type=int, default=1)
        parser.add_argument("--num-receivers", type=int, default=1)
        parser.add_argument("--vocab-size", type=int, default=5)    # Including the EOS token!
        parser.add_argument("--max-len", type=int, default=4)   # Excluding EOS token!
        parser.add_argument("--length-cost", type=float, default=0)   # Excluding EOS token!

        parser.add_argument("--noise", type=float, default=0)
        parser.add_argument("--feedback", default=False, action="store_true")
        parser.add_argument("--open-cr", default=False, action="store_true")

        parser.add_argument("--log-topsim-on-validation", default=False, action="store_true")
        parser.add_argument("--log-posdis-on-validation", default=False, action="store_true")
        parser.add_argument("--log-bosdis-on-validation", default=False, action="store_true")
        parser.add_argument("--log-entropy-on-validation", default=False, action="store_true")

        parser.add_argument("--sender_embed_dim", type=int, default=5)
        parser.add_argument("--sender-num-layers", type=int, default=1)
        parser.add_argument("--sender-hidden-dim", type=int, default=500)
        parser.add_argument("--sender-learning-speed", type=float, default=1)
        parser.add_argument("--sender-entropy-coeff", type=float, default=0.5)
        parser.add_argument("--sender-layer-norm", default=False, action="store_true")

        parser.add_argument("--receiver_embed_dim", type=int, default=30)
        parser.add_argument("--receiver-num-layers", type=int, default=1)
        parser.add_argument("--receiver-hidden-dim", type=int, default=500)
        parser.add_argument("--receiver-learning-speed", type=float, default=1)
        parser.add_argument("--receiver-entropy-coeff", type=float, default=0.5)
        parser.add_argument("--receiver-layer-norm", default=False, action="store_true")

        return parent_parser

    def init_agents(self):
        if self.params.symmetric:
            if self.params.num_senders != self.params.num_receivers:
                raise ValueError("Symmetric game requires same number of senders and receivers.")
            self.senders = ModuleList(
                [
                    SenderReceiver(self.params.num_attributes, self.params.num_values,
                                   self.params.vocab_size, self.params.sender_embed_dim,
                                   self.params.sender_hidden_dim, self.params.max_len,
                                   self.params.sender_layer_norm, self.params.receiver_layer_norm,
                                   self.params.sender_num_layers)
                    for _ in range(self.params.num_senders * 2)
                ]
            )
            self.receivers = self.senders
        else:
            if self.params.optimal_sender:
                self.senders = ModuleList(
                    [
                        OptimalSender(
                               self.params.num_attributes, self.params.num_values,
                               self.params.vocab_size, self.params.max_len)
                        for _ in range(self.params.num_senders)
                    ]
                )
            else:
                self.senders = ModuleList(
                    [
                        Sender(
                               self.params.num_attributes, self.params.num_values,
                               self.params.vocab_size, self.params.sender_embed_dim,
                               self.params.sender_hidden_dim, self.params.max_len,
                               self.params.sender_layer_norm, self.params.sender_num_layers,
                               self.params.feedback, self.params.open_cr)
                        for _ in range(self.params.num_senders)
                    ]
                )
            self.receivers = ModuleList(
                [
                    Receiver(self.params.vocab_size, self.params.receiver_embed_dim,
                             self.params.receiver_hidden_dim, self.params.max_len,
                             self.params.num_attributes, self.params.num_values,
                             self.params.receiver_layer_norm, self.params.receiver_num_layers,
                             self.params.open_cr)
                    for _ in range(self.params.num_receivers)
                ]
            )

    def init_MLP_receivers(self):
        self.receivers = ModuleList(
            [
                ReceiverMLP(self.params.vocab_size, self.params.receiver_embed_dim,
                            self.params.num_attributes, self.params.num_values,
                            self.params.max_len)
                for _ in range(self.params.num_receivers)
            ]
        )

    def freeze_senders(self):
        for sender in self.senders:
            for param in sender.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        if self.params.optimal_sender:
            optimizers_sender = []
        else:
            optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=1e-3) for sender in self.senders]
        optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=1e-3) for receiver in self.receivers]

        return tuple(itertools.chain(optimizers_sender, optimizers_receiver))

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        if self.params.symmetric:
            num_agents = self.params.num_senders + self.params.num_receivers
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
            sender_idx = random.choice(range(self.params.num_senders))
            receiver_idx = random.choice(range(self.params.num_receivers))

            if self.params.optimal_sender:
                opt_sender = None
                opts_receiver = optimizers
                if self.params.num_receivers == 1:
                    opt_receiver = opts_receiver
                else:
                    opt_receiver = opts_receiver[receiver_idx]

            else:
                opts_sender = optimizers[:self.params.num_senders]
                opts_receiver = optimizers[self.params.num_senders:]

                opt_sender = opts_sender[sender_idx]
                opt_receiver = opts_receiver[receiver_idx]

        if opt_sender:
            opt_sender.zero_grad()
        opt_receiver.zero_grad()
        loss, acc = self.forward(batch, sender_idx, receiver_idx)
        self.manual_backward(loss)

        perform_sender_update = torch.rand(1) < self.params.sender_learning_speed
        if perform_sender_update and opt_sender:
            opt_sender.step()

        perform_receiver_update = torch.rand(1) < self.params.receiver_learning_speed
        if perform_receiver_update:
            opt_receiver.step()

        self.log(f"train_acc", acc.float().mean(), prog_bar=True, add_dataloader_idx=False)

    def forward(
        self, sender_input, sender_idx, receiver_idx, return_messages=False, disable_noise=False
    ):
        sender = self.senders[sender_idx]
        receiver = self.receivers[receiver_idx]
        batch_size = sender_input.shape[0]

        messages_sender = []
        original_messages_sender = []
        sender_logits = []
        sender_entropies = []

        receiver_entropies = []
        receiver_hidden_states = torch.zeros((batch_size, self.params.max_len, self.params.receiver_hidden_dim)).type_as(sender_input)
        messages_receiver = []
        receiver_logits = []

        sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_first_turn(sender_input)
        sender_entropies.append(sender_step_entropy)
        sender_logits.append(sender_step_logits)
        original_messages_sender.append(sender_output_tokens)

        sender_output_tokens_detached = sender_output_tokens.detach().clone()
        sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)
        messages_sender.append(sender_output_tokens_detached)

        if self.params.feedback:
            receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward_first_turn(sender_output_tokens)
            receiver_entropies.append(receiver_step_entropy)
            receiver_logits.append(receiver_step_logits)
            receiver_hidden_states[:, 0] = receiver_prev_hidden[-1]
            input_cr = receiver_output_tokens.detach()
        else:
            input_cr = None

        for step in range(1, self.params.max_len):

            sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_subsequent_turn(sender_output_tokens, sender_prev_hidden, input_cr)

            sender_entropies.append(sender_step_entropy)
            sender_logits.append(sender_step_logits)
            original_messages_sender.append(sender_output_tokens)

            sender_output_tokens_detached = sender_output_tokens.detach().clone()
            sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)
            messages_sender.append(sender_output_tokens_detached)

            if self.params.feedback:
                receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward(sender_output_tokens_detached, receiver_prev_hidden)
                receiver_entropies.append(receiver_step_entropy)
                receiver_logits.append(receiver_step_logits)
                receiver_hidden_states[:, step] = receiver_prev_hidden[-1]
                messages_receiver.append(receiver_output_tokens)

                input_cr = receiver_output_tokens.detach()

        messages_sender = torch.stack(messages_sender).permute(1, 0)
        sender_logits = torch.stack(sender_logits).permute(1, 0)
        sender_entropies = torch.stack(sender_entropies).permute(1, 0)

        messages_sender_lengths = find_lengths(messages_sender)
        self.log(f"message_lengths", messages_sender_lengths.float().mean() - 1)

        if self.params.feedback:
            messages_receiver = torch.stack(messages_receiver).permute(1, 0)
            receiver_logits = torch.stack(receiver_logits).permute(1, 0)
            receiver_entropies = torch.stack(receiver_entropies).permute(1, 0)

            messages_receiver_lengths = find_lengths(messages_receiver, stop_at_eos=False)
            self.log(f"receiver_message_lengths", messages_receiver_lengths.float().mean() - 1)

        for i in range(messages_sender.size(1)):
            sender_entropies[i >= messages_sender_lengths, i] = 0
            sender_logits[i >= messages_sender_lengths, i] = 0
            receiver_hidden_states[i >= messages_sender_lengths, i] = 0
        effective_entropy_s_1 = sender_entropies.sum(dim=1) / messages_sender_lengths.float()
        effective_log_prob_s = sender_logits.sum(dim=1)

        entropy_loss = effective_entropy_s_1 * self.sender_entropy_coeff

        baseline = self.baselines["length_sender_1"].predict(messages_sender_lengths.device)
        policy_length_loss = (messages_sender_lengths.float() - baseline) * self.length_cost * effective_log_prob_s

        receiver_output = receiver.forward_output(messages_sender, messages_sender_lengths)

        receiver_output = receiver_output.view(batch_size, self.num_attributes, self.num_values)

        sender_input = sender_input.view(batch_size, self.num_attributes, self.num_values)

        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach(), dim=1) == self.num_attributes).float()

        receiver_output = receiver_output.view(batch_size * self.num_attributes, self.num_values)

        labels = sender_input.argmax(dim=-1).view(batch_size * self.num_attributes)
        receiver_loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.num_attributes).mean(dim=-1)

        self.log(f"receiver_loss", receiver_loss.mean())
        assert len(receiver_loss) == batch_size

        if self.params.feedback:
            for i in range(messages_receiver.size(1)):
                receiver_entropies[i >= messages_receiver_lengths, i] = 0
                receiver_logits[i >= messages_receiver_lengths, i] = 0
            effective_entropy_r = receiver_entropies.sum(dim=1) / messages_receiver_lengths.float()
            effective_log_prob_r = receiver_logits.sum(dim=1)

            entropy_loss = (effective_entropy_s_1 * self.sender_entropy_coeff
                                + effective_entropy_r * self.receiver_entropy_coeff)

            baseline = self.baselines["length_receiver_1"].predict(messages_receiver_lengths.device)
            policy_length_loss += (messages_receiver_lengths.float() - baseline) * self.length_cost * effective_log_prob_r

        self.log(f"entropy_loss", entropy_loss.mean())

        loss_baseline = self.baselines["loss"].predict(receiver_loss.device)
        policy_loss = (
            (receiver_loss.detach() - loss_baseline) * effective_log_prob_s
        )

        self.log(f"policy_loss", policy_loss.mean())
        self.log(f"policy_length_loss", policy_length_loss.mean())

        optimized_loss = policy_length_loss + policy_loss - entropy_loss

        # average over items in batch
        optimized_loss = optimized_loss.mean()

        # add the differentiable loss terms
        optimized_loss += receiver_loss.mean()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length_sender"].update(messages_sender_lengths.float())
            if self.params.feedback:
                self.baselines["length_receiver"].update(messages_receiver_lengths.float())

        if return_messages:
            return optimized_loss, acc, messages_sender
        else:
            return optimized_loss, acc

    def add_noise(self, messages, disable_noise=False):
        if not disable_noise and self.params.noise > 0:
            indices = torch.multinomial(torch.tensor([1 - self.params.noise, self.params.noise]),
                                        messages.numel(), replacement=True)
            indices = indices.reshape(messages.shape).to(messages.device)
            # Replace all randomly selected values (but only if they are not EOS symbols (0))
            messages[(indices == 1) & (messages != 0)] = self.token_noise
        return messages

    def on_validation_epoch_start(self):
        # Sample agent indices for this validation epoch
        if self.params.symmetric:
            num_agents = self.params.num_senders + self.params.num_receivers
            self.val_epoch_sender_idx = random.choice(range(num_agents))
            self.val_epoch_receiver_idx = random.choice(range(num_agents))
            # Avoid communication within same agent
            while (self.val_epoch_sender_idx == self.val_epoch_receiver_idx):
                self.val_epoch_sender_idx = random.choice(range(num_agents))
                self.val_epoch_receiver_idx = random.choice(range(num_agents))
        else:
            self.val_epoch_sender_idx = random.choice(range(self.params.num_senders))
            self.val_epoch_receiver_idx = random.choice(range(self.params.num_receivers))

        print(f"\nValidating for sender {self.val_epoch_sender_idx} and receiver {self.val_epoch_receiver_idx}:")

    def validation_step(self, sender_input, batch_idx, dataloader_idx):
        sender_idx = self.val_epoch_sender_idx
        receiver_idx = self.val_epoch_receiver_idx
        if dataloader_idx == 0:
            # Val Generalization
            _, acc = self.forward(sender_input, sender_idx, receiver_idx)
            _, acc_no_noise = self.forward(sender_input, sender_idx, receiver_idx, disable_noise=True)

            return acc, acc_no_noise

        elif dataloader_idx == 1:
            # Test Generalization
            _, acc = self.forward(sender_input, sender_idx, receiver_idx)
            _, acc_no_noise = self.forward(sender_input, sender_idx, receiver_idx, disable_noise=True)

            return acc, acc_no_noise

        elif dataloader_idx == 2:
            # Language analysis (on train set data)
            # TODO: do only if required for logging
            _, acc_no_noise, messages = self.forward(sender_input, sender_idx, receiver_idx, return_messages=True, disable_noise=True)

            return sender_input, messages, acc_no_noise

    def validation_epoch_end(self, validation_step_outputs):
        # Val Generalization:
        accs = torch.cat([acc for acc, _ in validation_step_outputs[0]])
        accs_no_noise = torch.cat([acc_no_noise for _, acc_no_noise in validation_step_outputs[0]])
        val_acc = accs.mean().item()
        val_acc_no_noise = accs_no_noise.mean().item()
        self.log("val_acc", val_acc, add_dataloader_idx=False)
        self.log("val_acc_no_noise", val_acc_no_noise, add_dataloader_idx=False)
        is_best_checkpoint = False
        if self.best_val_acc_no_noise < val_acc_no_noise:
            self.best_val_acc_no_noise = val_acc_no_noise
            is_best_checkpoint = True
        self.log("best_val_acc_no_noise", self.best_val_acc_no_noise, prog_bar=True, add_dataloader_idx=False)

        print("val_acc: ", val_acc)
        print("val_acc_no_noise: ", val_acc_no_noise)

        # Test Generalization:
        accs = torch.cat([acc for acc, _ in validation_step_outputs[1]])
        accs_no_noise = torch.cat([acc_no_noise for _, acc_no_noise in validation_step_outputs[1]])
        test_acc = accs.mean().item()
        test_acc_no_noise = accs_no_noise.mean().item()
        self.log("test_acc", test_acc, add_dataloader_idx=False)
        self.log("test_acc_no_noise", test_acc_no_noise, add_dataloader_idx=False)

        # Language analysis (on train set data)
        language_analysis_results = validation_step_outputs[2]
        train_acc_no_noise = torch.cat([acc for _, _, acc in language_analysis_results])
        self.log("train_acc_no_noise", train_acc_no_noise, prog_bar=True, add_dataloader_idx=False)
        if is_best_checkpoint:
            self.log("train_acc_no_noise_at_best_val_acc", train_acc_no_noise)

        meanings = torch.cat([meaning.cpu() for meaning, _, _ in language_analysis_results])
        messages = torch.cat([message.cpu() for _, message, _ in language_analysis_results])
        self.analyze_language(messages, meanings, is_best_checkpoint)

    def analyze_language(self, messages, meanings, is_best_checkpoint=False):
        num_unique_messages = len(messages.unique(dim=0))
        self.log("num_unique_messages", float(num_unique_messages))

        # TODO: command line arg:
        if is_best_checkpoint:
            meanings_strings = pd.DataFrame(meanings).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)
            num_digits = int(math.log10(self.params.vocab_size))
            messages_strings = pd.DataFrame(messages).apply(lambda row: "".join([s.zfill(num_digits) for s in row.astype(int).astype(str)]), axis=1)
            messages_df = pd.DataFrame([meanings_strings, messages_strings]).T
            messages_df.rename(columns={0: 'meaning', 1: 'message'}, inplace=True)
            messages_df.to_csv(f"{self.logger.log_dir}/messages.csv", index=False)

        if self.params.log_entropy_on_validation or is_best_checkpoint:
            entropy = compute_entropy(messages.numpy())
            self.log("message_entropy", entropy, prog_bar=True)
            print("message_entropy: ", entropy)
            if is_best_checkpoint:
                self.log("message_entropy_at_best_val_acc", entropy)

        if self.params.log_topsim_on_validation or is_best_checkpoint:
            topsim = compute_topsim(meanings, messages)
            self.log("topsim", topsim, prog_bar=True)
            print("Topsim: ", topsim)
            if is_best_checkpoint:
                self.log("topsim_at_best_val_acc", topsim)

        if self.params.log_posdis_on_validation or is_best_checkpoint:
            posdis = compute_posdis(self.num_attributes, self.num_values, meanings, messages)
            self.log("posdis", posdis, prog_bar=True)
            print("posdis: ", posdis)
            if is_best_checkpoint:
                self.log("posdis_at_best_val_acc", posdis)

        if self.params.log_bosdis_on_validation or is_best_checkpoint:
            bosdis = compute_bosdis(self.num_attributes, self.num_values, meanings, messages, self.params["vocab_size"])
            self.log("bosdis", bosdis, prog_bar=True)
            print("bodis: ", bosdis)
            if is_best_checkpoint:
                self.log("bosdis_at_best_val_acc", bosdis)

    def on_fit_start(self):
        # Set which metrics to use for hyperparameter tuning
        metrics = ["best_val_acc_no_noise", "topsim_at_best_val_acc", "posdis_at_best_val_acc",
                   "bosdis_at_best_val_acc", "test_acc_no_noise"]
        self.logger.log_hyperparams(self.hparams, {m: 0 for m in metrics})
