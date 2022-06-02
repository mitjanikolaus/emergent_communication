import itertools
import math
import random
from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn, jit
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.nn import ModuleList, Parameter

import pandas as pd

from data import get_speech_act, get_speech_act_codes, get_objects, REQUEST
from language_analysis import compute_topsim, compute_entropy
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


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, n_features, n_values, num_objects, speech_acts, layer_norm, num_layers
    ):
        super(Receiver, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_cell = LayerNormLSTMCell if layer_norm else nn.LSTMCell
        self.cells = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )

        self.fc1 = nn.Linear(n_features*n_values, hidden_size)

        self.speech_acts = speech_acts

        self.linear_out = nn.Linear(num_objects+1, num_objects+2)

        self.linear_speech_act = nn.Linear(hidden_size, 2)

    def forward(self, batch):
        message, input, message_lengths = batch
        batch_size = message.shape[0]
        num_objects = input.shape[1]
        embedded_message = self.embedding(message)

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)]
        prev_c = [
            torch.zeros((batch_size, self.hidden_size)).type_as(embedded_message) for _ in range(self.num_layers)
        ]

        max_message_len = embedded_message.shape[1]
        hidden_states = torch.zeros((batch_size, max_message_len, self.hidden_size)).type_as(embedded_message)
        for step in range(max_message_len):
            lstm_input = embedded_message[:, step]
            for i, layer in enumerate(self.cells):
                h_t, c_t = layer(lstm_input, (prev_hidden[i], prev_c[i]))
                prev_c[i] = c_t
                prev_hidden[i] = h_t
                lstm_input = h_t

            hidden_states[:, step] = h_t

        encoded_message = hidden_states[range(batch_size), message_lengths-1]

        embedded_input = self.fc1(input)

        embedded_input = embedded_input.tanh()

        product = torch.prod(embedded_input, dim=1).unsqueeze(1)

        catted = torch.cat((embedded_input, product), dim=1)

        output = torch.bmm(catted, torch.unsqueeze(encoded_message, dim=-1)).squeeze(2)

        output = self.linear_out(output)

        speech_act_out = self.linear_speech_act(encoded_message)
        softmax_speech_act = F.softmax(speech_act_out, dim=-1)

        speech_act_factor = torch.cat([softmax_speech_act[:, :1].repeat(1, num_objects), softmax_speech_act[:,1:].repeat(1, 2)], dim=1)
        output = speech_act_factor * output

        return output, speech_act_out


class Sender(pl.LightningModule):
    def __init__(
        self,
        speech_acts,
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

        self.speech_acts = speech_acts
        num_speech_acts = len(self.speech_acts)
        self.embed_input = nn.Linear(n_features*n_values+num_speech_acts, embed_dim)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        lstm_cell = LayerNormLSTMCell if layer_norm else nn.LSTMCell

        self.cells = nn.ModuleList(
            [
                lstm_cell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else lstm_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x):
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


class OptimalSender(pl.LightningModule):
    def __init__(
        self,
        speech_acts,
        n_features,
        n_values,
        vocab_size,
        max_len,
    ):
        super(OptimalSender, self).__init__()
        self.max_len = max_len

        self.speech_acts = speech_acts
        self.n_features = n_features
        self.n_values = n_values

        self.vocab_size = vocab_size

        assert len(speech_acts) <= n_values
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
        speech_act_codes_one_hot = get_speech_act_codes(intents, len(self.speech_acts))
        speech_act_codes = torch.nonzero(speech_act_codes_one_hot)[:, 1]
        speech_act_codes = speech_act_codes.unsqueeze(1)

        objects = get_objects(intents, len(self.speech_acts))
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
        self.num_objects = self.hparams["data"]["num_objects"]
        self.model_hparams = AttributeDict(self.hparams["model"])

        self.speech_acts = self.model_hparams.speech_acts
        self.init_agents()

        self.sender_entropy_coeff = self.model_hparams.sender_entropy_coeff
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

        self.automatic_optimization = False

    def init_agents(self):
        if self.model_hparams.symmetric:
            if self.model_hparams.num_senders != self.model_hparams.num_receivers:
                raise ValueError("Symmetric game requires same number of senders and receivers.")
            self.senders = ModuleList(
                [
                    SenderReceiver(self.model_hparams.speech_acts,
                        self.model_hparams.num_features, self.model_hparams.num_values,
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
                        OptimalSender(self.model_hparams.speech_acts,
                               self.model_hparams.num_features, self.model_hparams.num_values,
                               self.model_hparams.vocab_size, self.model_hparams.max_len)
                        for _ in range(self.model_hparams.num_senders)
                    ]
                )
            else:
                self.senders = ModuleList(
                    [
                        Sender(self.model_hparams.speech_acts,
                               self.model_hparams.num_features, self.model_hparams.num_values,
                               self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim,
                               self.model_hparams.sender_hidden_dim, self.model_hparams.max_len,
                               self.model_hparams.sender_layer_norm, self.model_hparams.sender_num_layers)
                        for _ in range(self.model_hparams.num_senders)
                    ]
                )
            self.init_receivers()

    def init_receivers(self):
        self.receivers = ModuleList(
            [
                Receiver(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim,
                         self.model_hparams.receiver_hidden_dim, self.model_hparams.num_features,
                         self.model_hparams.num_values, self.num_objects, self.model_hparams.speech_acts,
                         self.model_hparams.receiver_layer_norm, self.model_hparams.receiver_num_layers)
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

        # self.log(f"train_acc_sender_{sender_idx}_receiver_{receiver_idx}", acc, logger=True, add_dataloader_idx=False)
        self.log(f"train_acc", acc.float().mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        self.log_dict(get_acc_per_speech_act(batch, acc, self.speech_acts), prog_bar=True, logger=True, add_dataloader_idx=False)

        # self.log(f"train_loss", loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

    def forward(
        self, batch, sender_idx, receiver_idx, return_messages=False
    ):
        sender = self.senders[sender_idx]
        receiver = self.receivers[receiver_idx]

        sender_input, receiver_input, labels = batch
        messages, log_prob_s, entropy_s = sender(sender_input)
        message_lengths = find_lengths(messages)
        self.log(f"message_lengths", message_lengths.type(torch.float).mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        receiver_output, receiver_out_speech_act = receiver(
            (messages, receiver_input, message_lengths)
        )

        acc = (receiver_output.argmax(dim=1) == labels).detach()
        batch_size = sender_input.shape[0]
        receiver_loss = F.cross_entropy(receiver_output, labels, reduction='none')
        self.log(f"receiver_loss", receiver_loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        if self.model_hparams["receiver_aux_loss"]:
            labels_speech_act = labels.clone().detach() >= self.num_objects
            labels_speech_act = labels_speech_act.type(torch.long)
            acc_speech_act = (receiver_out_speech_act.argmax(dim=1) == labels_speech_act).detach()
            receiver_speech_act_loss = F.cross_entropy(receiver_out_speech_act, labels_speech_act, reduction='none')
            self.log(f"receiver_speech_act_loss", receiver_speech_act_loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)
            self.log(f"train_acc_speech_act", acc_speech_act.float().mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

            receiver_loss += receiver_speech_act_loss

        assert len(receiver_loss) == batch_size

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros(batch_size).type_as(sender_input)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros(batch_size).type_as(sender_input)

        for i in range(messages.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
        )

        self.log(f"effective_log_prob_s", effective_log_prob_s.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f"weighted_entropy", weighted_entropy.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        log_prob = effective_log_prob_s

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = (
            (length_loss - self.baselines["length"].predict(length_loss))
            * effective_log_prob_s
        ).mean()
        loss_baseline = self.baselines["loss"].predict(receiver_loss.detach())
        policy_loss = (
            (receiver_loss.detach() - loss_baseline) * log_prob
        ).mean()

        self.log(f"policy_loss", policy_loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f"policy_length_loss", policy_length_loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # add the receiver loss
        optimized_loss += receiver_loss.mean()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length"].update(length_loss)

        if return_messages:
            return optimized_loss, acc, messages
        else:
            return optimized_loss, acc

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # Generalization
            sender_idx = self.val_epoch_sender_idx
            receiver_idx = self.val_epoch_receiver_idx

            _, acc, messages = self.forward(batch, sender_idx, receiver_idx, return_messages=True)
            return get_acc_per_speech_act(batch, acc, self.speech_acts, is_test_acc=True), messages
        elif dataloader_idx == 1:
            # Language analysis
            sender_idx = self.val_epoch_sender_idx
            sender = self.senders[sender_idx]

            sender_input = batch
            messages, _, _ = sender(sender_input)
            return sender_input, messages

    def validation_epoch_end(self, validation_step_outputs):
        # Generalization:
        generalization_results = validation_step_outputs[0]
        accs = [acc for acc, _ in generalization_results]
        generalization_results = pd.DataFrame.from_records(accs)
        test_acc = generalization_results.mean().to_dict()
        self.log_dict(test_acc, prog_bar=True, logger=True, add_dataloader_idx=False)
        print("test_acc: ", test_acc)

        # Language analysis
        language_analysis_results = validation_step_outputs[1]
        self.analyze_language(language_analysis_results)

    def analyze_language(self, lang_analysis_results):
        meanings = torch.cat([meaning for meaning, _ in lang_analysis_results])
        messages = torch.cat([message for _, message in lang_analysis_results])
        speech_acts = np.array([get_speech_act(intent, self.speech_acts) for intent in meanings])

        num_unique_messages = len(messages.unique(dim=0))
        self.log("num_unique_messages", float(num_unique_messages), prog_bar=True, logger=True)

        meanings_strings = pd.DataFrame(meanings).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)

        num_digits = int(math.log10(self.model_hparams.vocab_size))
        messages_strings = pd.DataFrame(messages).apply(lambda row: "".join([s.zfill(num_digits) for s in row.astype(int).astype(str)]), axis=1)
        messages_df = pd.DataFrame([meanings_strings, messages_strings]).T
        messages_df.rename(columns={0: 'meaning', 1: 'message'}, inplace=True)
        messages_df.to_csv(f"{self.logger.log_dir}/messages.csv", index=False)

        meanings_request = meanings[speech_acts == REQUEST].cpu()
        messages_request = messages[speech_acts == REQUEST].cpu()
        if self.model_hparams.log_entropy_on_validation:
            entropy = compute_entropy(messages_request)
            self.log("message_entropy", entropy, prog_bar=True, logger=True)
            print("message_entropy: ", entropy)

        if self.model_hparams.log_topsim_on_validation:
            topsim = compute_topsim(meanings_request, messages_request)
            self.log("topsim", topsim, prog_bar=True, logger=True)
            print("Topsim: ", topsim)

    def on_fit_start(self):
        # Set which metrics to use for hyperparameter tuning
        metrics = self.speech_acts.copy()
        for speech_act in self.speech_acts:
            metrics.append(speech_act + "_test")

        metrics.append("topsim")
        self.logger.log_hyperparams(self.hparams, {m: 0 for m in metrics})


def get_acc_per_speech_act(batch, acc, speech_acts, is_test_acc=False):
    sender_input, _, _ = batch
    accs = {}
    speech_acts_batch = np.array([get_speech_act(intent, speech_acts) for intent in sender_input])
    for speech_act in speech_acts:
        name = speech_act
        if is_test_acc:
            name += "_test"
        accs[name] = acc[speech_acts_batch == speech_act].float().mean().item()
    return accs
