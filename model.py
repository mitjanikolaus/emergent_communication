from collections import defaultdict

import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl

from utils import MeanBaseline, find_lengths, NoBaseline


class Receiver(nn.Module):
    def __init__(
            self, vocab_size, embed_dim, hidden_size, n_features, num_layers=1
    ):
        super(Receiver, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.fc1 = nn.Linear(n_features, hidden_size)

    def forward(self, message, input=None, lengths=None):
        # print("\n\nMessage: ", message)
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (rnn_hidden, _) = self.lstm(packed)

        #TODO:
        # encoded = out?
        encoded_message = rnn_hidden[-1]

        embedded_input = self.fc1(input).tanh()
        dots = torch.matmul(embedded_input, torch.unsqueeze(encoded_message, dim=-1)).squeeze(2)
        softmaxed = F.softmax(dots, dim=1)
        return softmaxed


class Sender(nn.Module):
    def __init__(
        self,
        n_features,
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

        self.embed_input = nn.Linear(n_features, hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

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
        #TODO: layernorm?

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = [self.embed_input(x)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )
        batch_size = x.shape[0]

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]

        input = torch.stack([self.sos_embedding] * batch_size)

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

        self.sender = Sender(self.model_hparams.num_features, self.model_hparams.vocab_size, self.model_hparams.sender_embed_dim, self.model_hparams.sender_hidden_dim, self.model_hparams.max_len, self.model_hparams.sender_num_layers)
        self.receiver = Receiver(self.model_hparams.vocab_size, self.model_hparams.receiver_embed_dim, self.model_hparams.receiver_hidden_dim, self.model_hparams.num_features, self.model_hparams.receiver_num_layers)

        self.sender_entropy_coeff = self.model_hparams.sender_entropy_coeff
        self.length_cost = self.model_hparams.length_cost

        if self.model_hparams.baseline_type == "mean":
            self.baselines = defaultdict(MeanBaseline)
        elif self.model_hparams.baseline_type == "none":
            self.baselines = defaultdict(NoBaseline)
        else:
            raise ValueError("Unknown baseline type: ", self.model_hparams.baseline_type)

    def training_step(self, batch, batch_idx):
        loss, interactions = self.forward(batch)
        return loss

    def configure_optimizers(self):
        # TODO separate optimizers for sender and receiver?
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(
        self, batch,
    ):
        sender_input, receiver_input, labels = batch
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_length = find_lengths(message)
        receiver_output = self.receiver(
            message, receiver_input, message_length
        )

        acc = (receiver_output.argmax(dim=1) == labels).detach().float().mean()
        self.log("accuracy", acc, on_step=True, prog_bar=True, logger=True)

        batch_size = sender_input.shape[0]
        receiver_loss = F.cross_entropy(receiver_output, labels, reduction='none')
        assert len(receiver_loss) == batch_size
        # loss, aux_info = loss(
        #     sender_input, message, receiver_input, receiver_output, labels
        # )
        self.log("cross_entropy_loss", receiver_loss.mean(), on_step=True, prog_bar=True, logger=True)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros(batch_size)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros(batch_size)

        for i in range(message.size(1)):
            not_eosed = (i < message_length).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_length.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
        )
        self.log("weighted_entropy", weighted_entropy, on_step=True, prog_bar=True, logger=True)

        log_prob = effective_log_prob_s

        length_loss = message_length.float() * self.length_cost

        policy_length_loss = (
            (length_loss - self.baselines["length"].predict(length_loss))
            * effective_log_prob_s
        ).mean()
        loss_baseline = self.baselines["loss"].predict(receiver_loss.detach())
        policy_loss = (
            (receiver_loss.detach() - loss_baseline) * log_prob
        ).mean()

        self.log("policy_length_loss", policy_length_loss, on_step=True, prog_bar=True, logger=True)
        self.log("loss_baseline", loss_baseline, on_step=True, prog_bar=True, logger=True)
        self.log("policy_loss", policy_loss, on_step=True, prog_bar=True, logger=True)

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # add the receiver loss
        optimized_loss += receiver_loss.mean()

        if self.training:
            self.baselines["loss"].update(receiver_loss)
            self.baselines["length"].update(length_loss)

        interactions = dict(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            sender_entropy=entropy_s.detach(),
            length=message_length.float()
        )

        return optimized_loss, interactions

    # def optimizer_step(
    #         self,
    #         epoch,
    #         batch_idx,
    #         optimizer,
    #         optimizer_idx,
    #         optimizer_closure,
    #         on_tpu=False,
    #         using_native_amp=False,
    #         using_lbfgs=False,
    # ):
    #     # TODO separate optimizers for sender and receiver?
    #     # update generator every step
    #     optimizer.step(closure=optimizer_closure)
