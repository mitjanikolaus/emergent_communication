from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl

import pandas as pd

from language_analysis import compute_topsim, compute_entropy
from utils import MeanBaseline, find_lengths, NoBaseline


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, n_features, n_values, num_layers=1
    ):
        super(Receiver, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.fc1 = nn.Linear(n_features*n_values, hidden_size)

    def forward(self, message, input=None, lengths=None):
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (rnn_hidden, _) = self.lstm(packed)

        encoded_message = rnn_hidden[-1]

        embedded_input = self.fc1(input).tanh()
        dots = torch.matmul(embedded_input, torch.unsqueeze(encoded_message, dim=-1)).squeeze(2)
        softmaxed = F.softmax(dots, dim=1)
        return softmaxed


class Sender(nn.Module):
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
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        """
        super(Sender, self).__init__()
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

    def forward(self, x):
        batch_size = x.shape[0]

        prev_hidden = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

        prev_c = [
            torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)
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

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class SignalingGameModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_hparams = AttributeDict(self.hparams["model"])

        self.sender = Sender(self.model_hparams.num_features, self.model_hparams.num_values, self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim, self.model_hparams.sender_hidden_dim, self.model_hparams.max_len, self.model_hparams.sender_num_layers)
        self.receiver = Receiver(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim, self.model_hparams.receiver_hidden_dim, self.model_hparams.num_features, self.model_hparams.num_values, self.model_hparams.receiver_num_layers)

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

    def configure_optimizers(self):
        optimizer_sender = torch.optim.Adam(self.sender.parameters(), lr=1e-3)
        optimizer_receiver = torch.optim.Adam(self.receiver.parameters(), lr=1e-3)

        return optimizer_sender, optimizer_receiver

    def training_step(self, batch, batch_idx):
        opt_sender, opt_receiver = self.optimizers()
        opt_sender.zero_grad()
        opt_receiver.zero_grad()
        loss, acc = self.forward(batch)
        self.manual_backward(loss)

        perform_sender_update = torch.rand(1) < self.model_hparams.sender_learning_speed
        if perform_sender_update:
            opt_sender.step()

        perform_receiver_update = torch.rand(1) < self.model_hparams.receiver_learning_speed
        if perform_receiver_update:
            opt_receiver.step()

    def forward(
        self, batch,
    ):
        sender_input, receiver_input, labels = batch
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)
        receiver_output = self.receiver(
            message, receiver_input, message_lengths
        )

        acc = (receiver_output.argmax(dim=1) == labels).detach().float().mean()
        self.log("train_acc", acc, prog_bar=True, logger=True, add_dataloader_idx=False)

        batch_size = sender_input.shape[0]
        receiver_loss = F.cross_entropy(receiver_output, labels, reduction='none')
        assert len(receiver_loss) == batch_size
        self.log("receiver_loss", receiver_loss.mean(), prog_bar=True, logger=True, add_dataloader_idx=False)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros(batch_size)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros(batch_size)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
        )
        self.log("weighted_entropy", weighted_entropy, prog_bar=True, logger=True, add_dataloader_idx=False)

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

        self.log("policy_length_loss", policy_length_loss, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log("loss_baseline", loss_baseline, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log("policy_loss", policy_loss, prog_bar=True, logger=True, add_dataloader_idx=False)

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # add the receiver loss
        optimized_loss += receiver_loss.mean()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length"].update(length_loss)

        return optimized_loss, acc

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # Generalization:
            loss, acc = self.forward(batch)
            return acc
        else:
            # Language analysis
            messages, log_prob_s, entropy_s = self.sender(batch)
            return batch, messages

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
        messages_strings = pd.DataFrame(messages).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)

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
