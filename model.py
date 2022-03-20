import itertools
import math
import random
from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.nn import ModuleList


import pandas as pd

from data import SPEECH_ACTS, get_speech_act
from language_analysis import compute_topsim, compute_entropy
from utils import MeanBaseline, find_lengths, NoBaseline


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, n_features, n_values, n_distractors, num_layers=1
    ):
        super(Receiver, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.lstm_speech_act = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.fc1 = nn.Linear(n_features*n_values, hidden_size)

        self.output_layer = nn.Linear(hidden_size*2, n_distractors+2)


    def forward(self, batch):
        message, input, message_lengths = batch
        batch_size = message.shape[0]
        emb = self.embedding(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, message_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (rnn_hidden, _) = self.lstm(packed)
        encoded_message = rnn_hidden[-1]

        embedded_input = self.fc1(input)
        embedded_input = embedded_input.tanh()

        dots = torch.matmul(embedded_input, torch.unsqueeze(encoded_message, dim=-1)).squeeze(2)

        _, (rnn_hidden_speech_act, _) = self.lstm_speech_act(packed)
        encoded_message_speech_act = rnn_hidden_speech_act[-1]

        dots_2 = torch.matmul(dots.unsqueeze(2), encoded_message_speech_act.unsqueeze(1)).reshape(batch_size, -1)

        out = self.output_layer(dots_2)
        softmaxed = F.softmax(out, dim=1)
        return softmaxed


class Sender(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_values,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
    ):
        super(Sender, self).__init__()
        self.max_len = max_len

        num_speech_acts = len(SPEECH_ACTS)
        self.embed_input = nn.Linear(n_features*n_values+num_speech_acts, embed_dim)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

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

            # TODO: use actual layer norm LSTM cell
            h_t = self.layer_norm(h_t)

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


class SenderReceiver(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_values,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
    ):
        super(SenderReceiver, self).__init__()
        self.max_len = max_len

        self.embed_input = nn.Linear(n_features*n_values, embed_dim)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

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

            # TODO: use actual layer norm LSTM cell
            h_t = self.layer_norm(h_t)

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
        for distractor_idx in range(input_receiver.shape[1]):
            input = self.embed_input(input_receiver[:,distractor_idx])

            prev_hidden = [torch.zeros((batch_size, self.hidden_size)).type_as(input_receiver) for _ in
                           range(self.num_layers)]

            prev_c = [
                torch.zeros((batch_size, self.hidden_size)).type_as(input_receiver) for _ in range(self.num_layers)
            ]


            perplexities_distractor = []

            for step in range(self.max_len):
                for i, layer in enumerate(self.cells):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                    prev_hidden[i] = h_t
                    input = h_t

                # TODO: use actual layer norm LSTM cell
                h_t = self.layer_norm(h_t)

                step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
                # TODO: EOS handling?
                ppls = step_logits[range(batch_size), message[:,step]]
                perplexities_distractor.append(ppls)

                x = message[:, step]
                input = self.embedding(x)

            perplexities_distractor = torch.stack(perplexities_distractor).sum(dim=0)
            perplexities.append(perplexities_distractor)

        perplexities = torch.stack(perplexities).permute(1,0)

        softmaxed = F.softmax(perplexities, dim=1)

        return softmaxed


class SignalingGameModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_distractors = self.hparams["data"]["num_distractors"]
        self.model_hparams = AttributeDict(self.hparams["model"])

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
                    SenderReceiver(self.model_hparams.num_features, self.model_hparams.num_values,
                        self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim,
                        self.model_hparams.sender_hidden_dim, self.model_hparams.max_len,
                        self.model_hparams.sender_num_layers)
                    for _ in range(self.model_hparams.num_senders * 2)
                ]
            )
            self.receivers = self.senders
        else:
            self.senders = ModuleList(
                [
                    Sender(self.model_hparams.num_features, self.model_hparams.num_values,
                        self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim,
                        self.model_hparams.sender_hidden_dim, self.model_hparams.max_len,
                        self.model_hparams.sender_num_layers)
                    for _ in range(self.model_hparams.num_senders)
                ]
            )

            self.receivers = ModuleList(
                [
                    Receiver(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim,
                                    self.model_hparams.receiver_hidden_dim, self.model_hparams.num_features,
                                    self.model_hparams.num_values, self.num_distractors,
                                    self.model_hparams.receiver_num_layers)
                    for _ in range(self.model_hparams.num_receivers)
                 ]
            )

    def configure_optimizers(self):
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
            opts_sender = optimizers[:self.model_hparams.num_senders]
            opts_receiver = optimizers[self.model_hparams.num_senders:]

            # Sample sender and receiver for this batch:
            sender_idx = random.choice(range(self.model_hparams.num_senders))
            receiver_idx = random.choice(range(self.model_hparams.num_receivers))

            opt_sender = opts_sender[sender_idx]
            opt_receiver = opts_receiver[receiver_idx]

        opt_sender.zero_grad()
        opt_receiver.zero_grad()
        loss, acc = self.forward(batch, sender_idx, receiver_idx)
        self.manual_backward(loss)

        perform_sender_update = torch.rand(1) < self.model_hparams.sender_learning_speed
        if perform_sender_update:
            opt_sender.step()

        perform_receiver_update = torch.rand(1) < self.model_hparams.receiver_learning_speed
        if perform_receiver_update:
            opt_receiver.step()

        # self.log(f"train_acc_sender_{sender_idx}_receiver_{receiver_idx}", acc, logger=True, add_dataloader_idx=False)
        self.log(f"train_acc", acc.float().mean(), prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f"speech_act_acc", get_acc_per_speech_act(batch, acc), prog_bar=True, logger=True, add_dataloader_idx=False)

        self.log(f"train_loss", loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

    def forward(
        self, batch, sender_idx, receiver_idx
    ):
        sender = self.senders[sender_idx]
        receiver = self.receivers[receiver_idx]

        sender_input, receiver_input, labels = batch
        message, log_prob_s, entropy_s = sender(sender_input)
        message_lengths = find_lengths(message)
        receiver_output = receiver(
            (message, receiver_input, message_lengths)
        )

        acc = (receiver_output.argmax(dim=1) == labels).detach()

        batch_size = sender_input.shape[0]
        receiver_loss = F.cross_entropy(receiver_output, labels, reduction='none')
        assert len(receiver_loss) == batch_size

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros(batch_size).type_as(sender_input)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros(batch_size).type_as(sender_input)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
        )

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

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # add the receiver loss
        optimized_loss += receiver_loss.mean()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length"].update(length_loss)

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
        sender_idx = self.val_epoch_sender_idx
        receiver_idx = self.val_epoch_receiver_idx

        if dataloader_idx == 0:
            # Generalization:
            loss, acc = self.forward(batch, sender_idx, receiver_idx)
            return acc.float().mean().cpu()
        else:
            # Language analysis
            sender = self.senders[sender_idx]
            sender_input, _, _ = batch
            messages, log_prob_s, entropy_s = sender(sender_input)
            return sender_input.cpu(), messages.cpu()

    def validation_epoch_end(self, validation_step_outputs):
        generalization_results, lang_analysis_results = validation_step_outputs

        # Generalization:
        test_acc = np.mean(generalization_results).item()
        self.log("test_acc", test_acc, prog_bar=True, logger=True, add_dataloader_idx=False)
        print("test_acc: ", test_acc)

        # Language analysis
        self.analyze_language(lang_analysis_results)

    def analyze_language(self, lang_analysis_results):
        meanings = torch.cat([meaning for meaning, message in lang_analysis_results])
        messages = torch.cat([message for meaning, message in lang_analysis_results])

        meanings_strings = pd.DataFrame(meanings).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)

        num_digits = int(math.log10(self.model_hparams.vocab_size))+1
        messages_strings = pd.DataFrame(messages).apply(lambda row: "".join([s.zfill(num_digits) for s in row.astype(int).astype(str)]), axis=1)
        messages_df = pd.DataFrame([meanings_strings, messages_strings]).T
        messages_df.rename(columns={0: 'meaning', 1: 'message'}, inplace=True)
        messages_df.to_csv(f"{self.logger.log_dir}/messages.csv", index=False)

        num_unique_messages = len(messages_strings.unique())
        self.log("num_unique_messages", float(num_unique_messages), prog_bar=True, logger=True)
        print("num_unique_messages: ", num_unique_messages)

        if self.model_hparams.log_entropy_on_validation:
            entropy = compute_entropy(messages)
            self.log("message_entropy", entropy, prog_bar=True, logger=True)
            print("message_entropy: ", entropy)

        if self.model_hparams.log_topsim_on_validation:
            topsim = compute_topsim(meanings, messages)
            self.log("topsim", topsim, prog_bar=True, logger=True)
            print("Topsim: ", topsim)


def get_acc_per_speech_act(batch, acc):
    sender_input, _, _ = batch
    accs = {}
    speech_acts_batch = np.array([get_speech_act(intent) for intent in sender_input])
    for speech_act in SPEECH_ACTS:
        accs[speech_act] = acc[speech_acts_batch == speech_act].float().mean()
    return accs