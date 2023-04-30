import itertools
import math
import random
from collections import defaultdict

import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.nn import ModuleList, GRUCell

from language_analysis import compute_topsim, compute_entropy, compute_posdis, compute_bosdis
from utils import RESNET_IMG_FEATS_DIM
from utils import MeanBaseline, find_lengths, NoBaseline


MAX_SAMPLES_LANGUAGE_ANALYSIS = 1000


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
            self, vocab_size, embed_dim, hidden_size, max_len, input_size, layer_norm, num_layers,
            feedback, vocab_size_feedback, object_attention, output_attention, discrimination_num_objects,
            noise=False
    ):
        super(Receiver, self).__init__()

        self.vocab_size = vocab_size
        vocab_size_perception = vocab_size
        # Add one symbol for noise treatment
        if noise:
            vocab_size_perception = vocab_size + 1

        self.embedding_sender = nn.Embedding(vocab_size_perception, embed_dim)
        self.embedding_receiver = nn.Embedding(vocab_size_feedback, embed_dim)

        self.sos_embedding_perc = nn.Parameter(torch.zeros(embed_dim))

        self.linear_objects_in = nn.Linear(input_size, hidden_size)
        self.attn_obj = nn.Linear(hidden_size, hidden_size * discrimination_num_objects)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len

        self.feedback = feedback

        self.object_attention = object_attention
        self.output_attention = output_attention

        if layer_norm:
            rnn_cell = LayerNormGRUCell
        else:
            rnn_cell = GRUCell

        rnn_input_size = embed_dim
        if self.feedback:
            rnn_input_size += embed_dim

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.cells = nn.ModuleList(
            [
                rnn_cell(input_size=rnn_input_size, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(num_layers)
            ]
        )

        self.attn = nn.Linear(hidden_size, max_len * hidden_size)
        self.hidden_to_objects_mul = nn.Linear(hidden_size, hidden_size)

        self.hidden_to_feedback_output = nn.Linear(hidden_size, vocab_size_feedback)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for module in layer:
                    module.reset_parameters()

    def forward_first_turn(self, candidate_objects, sender_messages=None):
        batch_size = candidate_objects.shape[0]

        embedded_objects = self.linear_objects_in(candidate_objects)
        embedded_objects_avg = torch.mean(embedded_objects, dim=1)

        if self.object_attention:
            attn_weights = F.softmax(
                self.attn_obj(embedded_objects_avg).reshape(batch_size, -1, embedded_objects.shape[-1]), dim=1)
            embedded_objects_transformed = torch.sum(attn_weights * embedded_objects, dim=1)
        else:
            embedded_objects_transformed = embedded_objects_avg

        prev_hidden = [embedded_objects_transformed]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        if sender_messages is None:
            messages_embedded = torch.stack([self.sos_embedding_perc] * batch_size)
        else:
            messages_embedded = self.embedding_sender(sender_messages)

        if self.feedback:
            prev_receiver_messages = torch.stack([self.sos_embedding] * batch_size)

            return self.forward(messages_embedded, prev_hidden, prev_receiver_messages)
        else:
            return self.forward(messages_embedded, prev_hidden)

    def forward_subsequent_turn(self, sender_messages, prev_hidden, prev_receiver_messages=None):
        messages_embedded = self.embedding_sender(sender_messages)
        if self.feedback:
            prev_msg_embedding = self.embedding_receiver(prev_receiver_messages)
            return self.forward(messages_embedded, prev_hidden, prev_msg_embedding)
        else:
            return self.forward(messages_embedded, prev_hidden)

    def forward(self, sender_messages_embedded, prev_hidden, prev_msg_embedding=None):
        rnn_input = sender_messages_embedded

        if self.feedback:
            rnn_input = torch.cat((sender_messages_embedded, prev_msg_embedding), dim=-1)

        for i, layer in enumerate(self.cells):
            h_t = layer(rnn_input, prev_hidden[i])
            prev_hidden[i] = h_t
            rnn_input = h_t

        if self.feedback:
            step_probs = F.softmax(self.hidden_to_feedback_output(h_t), dim=1)
            distr = Categorical(probs=step_probs)

            if self.training:
                output_token = distr.sample()
            else:
                output_token = step_probs.argmax(dim=1)

            entropy = distr.entropy()
            logits = distr.log_prob(output_token)
        else:
            output_token = None
            entropy = None
            logits = None

        return output_token, entropy, logits, prev_hidden

    def output(self, candidate_objects, hidden_states, message_lengths):
        batch_size = hidden_states.shape[0]

        embedded_objects = self.linear_objects_in(candidate_objects)

        if self.output_attention:
            hidden_states = self.hidden_to_objects_mul(hidden_states)

            for i in range(self.max_len):
                hidden_states[i >= message_lengths, i] = 0

            hidden_states_summary = torch.mean(hidden_states, dim=1)
            attn_weights = F.softmax(self.attn(hidden_states_summary).reshape(batch_size, self.max_len, -1), dim=1)

            hidden_states_transformed = torch.sum(hidden_states * attn_weights, dim=1)
        else:
            last_hidden_states = hidden_states[range(batch_size), message_lengths-1]
            hidden_states_transformed = self.hidden_to_objects_mul(last_hidden_states)

        output = torch.matmul(embedded_objects, hidden_states_transformed.unsqueeze(2))

        output = output.squeeze()

        return output


class Sender(pl.LightningModule):
    def __init__(
        self,
        input_size,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        layer_norm,
        num_layers,
        feedback,
        vocab_size_feedback,
    ):
        super(Sender, self).__init__()

        self.max_len = max_len

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.sos_embedding_perc = nn.Parameter(torch.zeros(embed_dim))

        self.linear_in_objects = nn.Linear(input_size, hidden_size)

        self.feedback = feedback

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding_prod = nn.Embedding(vocab_size, embed_dim)

        self.embedding_response = nn.Embedding(vocab_size_feedback, embed_dim)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        if layer_norm:
            rnn_cell = LayerNormGRUCell
        else:
            rnn_cell = GRUCell

        rnn_input_size = embed_dim
        if self.feedback:
            rnn_input_size += embed_dim

        self.cells = nn.ModuleList(
            [
                rnn_cell(input_size=rnn_input_size, hidden_size=hidden_size)
                if i == 0
                else rnn_cell(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for module in layer:
                    module.reset_parameters()

    def forward_first_turn(self, input_objects, receiver_message=None):
        batch_size = input_objects.shape[0]

        prev_hidden = [self.linear_in_objects(input_objects)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_msg_embedding = torch.stack([self.sos_embedding] * batch_size)

        if self.feedback:
            if receiver_message is None:
                embedded_receiver_message = torch.stack([self.sos_embedding_perc] * batch_size)
            else:
                embedded_receiver_message = self.embedding_response(receiver_message)

            return self.forward(prev_msg_embedding, prev_hidden, embedded_receiver_message)
        else:
            return self.forward(prev_msg_embedding, prev_hidden)

    def forward_subsequent_turn(self, output_last_turn, prev_hidden, receiver_message):
        embedded_output_last_turn = self.embedding_prod(output_last_turn)

        if self.feedback:
            embedded_receiver_message = self.embedding_response(receiver_message)
            return self.forward(embedded_output_last_turn, prev_hidden, embedded_receiver_message)
        else:
            return self.forward(embedded_output_last_turn, prev_hidden)

    def forward(self, prev_msg_embedding, prev_hidden, embedded_receiver_message=None):
        input = prev_msg_embedding
        if self.feedback:
            input = torch.cat((prev_msg_embedding, embedded_receiver_message), dim=1)

        for i, layer in enumerate(self.cells):
            h_t = layer(input, prev_hidden[i])
            prev_hidden[i] = h_t
            input = h_t

        step_probs = F.softmax(self.hidden_to_output(h_t), dim=1)
        distr = Categorical(probs=step_probs)

        if self.training:
            output_tokens = distr.sample()
        else:
            output_tokens = step_probs.argmax(dim=1)

        entropy = distr.entropy()
        logits = distr.log_prob(output_tokens)

        return output_tokens, entropy, logits, prev_hidden


class SignalingGameModule(pl.LightningModule):
    def __init__(self, symmetric=False, optimal_sender=False, load_checkpoint=None, baseline_type="mean",
                 max_len=4, length_cost=0, num_attributes=4, num_values=4, num_senders=1, num_receivers=1,
                 receiver_embed_dim=30, receiver_num_layers=500, receiver_hidden_dim=500,
                 receiver_learning_speed=1, sender_embed_dim=5, sender_entropy_coeff=0.5,
                 receiver_entropy_coeff=0.5, sender_num_layers=1, receiver_layer_norm=False,
                 sender_layer_norm=False, sender_hidden_dim=500, sender_learning_speed=1,
                 vocab_size=5, noise=0, feedback=False, vocab_size_feedback=3,
                 log_topsim_on_validation=False, log_posdis_on_validation=False,
                 log_bosdis_on_validation=False, log_entropy_on_validation=False,
                 guesswhat=False, imagenet=False, **kwargs):
        super().__init__()

        self.input_size = num_attributes * num_values
        if guesswhat or imagenet:
            self.input_size = RESNET_IMG_FEATS_DIM

        self.save_hyperparameters()
        self.params = AttributeDict(self.hparams)

        self.guesswhat = guesswhat
        self.imagenet = imagenet

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
        self.force_log = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--discrimination-num-objects", type=int, default=2)
        parser.add_argument("--hard-distractors", default=False, action="store_true")

        parser.add_argument("--receiver-starts", default=False, action="store_true")

        parser.add_argument("--initial-lr", type=float, default=1e-3)

        parser.add_argument("--guesswhat", default=False, action="store_true")
        parser.add_argument("--imagenet", default=False, action="store_true")

        parser.add_argument("--symmetric", default=False, action="store_true")
        parser.add_argument("--optimal-sender", default=False, action="store_true")
        parser.add_argument("--load-checkpoint", type=str, default=None)
        parser.add_argument("--baseline-type", type=str, default="mean")

        parser.add_argument("--num-attributes", type=int, default=4)
        parser.add_argument("--num-values", type=int, default=100)
        parser.add_argument("--num-senders", type=int, default=1)
        parser.add_argument("--num-receivers", type=int, default=1)
        parser.add_argument("--vocab-size", type=int, default=101)    # Including the EOS token!
        parser.add_argument("--vocab-size-feedback", type=int, default=100)
        parser.add_argument("--max-len", type=int, default=4)   # Excluding EOS token!
        parser.add_argument("--length-cost", type=float, default=0)   # Excluding EOS token!

        parser.add_argument("--noise", type=float, default=0)
        parser.add_argument("--feedback", default=False, action="store_true")

        parser.add_argument("--log-topsim-on-validation", default=False, action="store_true")
        parser.add_argument("--log-posdis-on-validation", default=False, action="store_true")
        parser.add_argument("--log-bosdis-on-validation", default=False, action="store_true")
        parser.add_argument("--log-entropy-on-validation", default=False, action="store_true")

        parser.add_argument("--sender-embed-dim", type=int, default=16)
        parser.add_argument("--sender-num-layers", type=int, default=1)
        parser.add_argument("--sender-hidden-dim", type=int, default=256)
        parser.add_argument("--sender-learning-speed", type=float, default=1)
        parser.add_argument("--sender-entropy-coeff", type=float, default=0.5)
        parser.add_argument("--sender-layer-norm", default=False, action="store_true")

        parser.add_argument("--receiver-embed-dim", type=int, default=16)
        parser.add_argument("--receiver-num-layers", type=int, default=1)
        parser.add_argument("--receiver-hidden-dim", type=int, default=256)
        parser.add_argument("--receiver-learning-speed", type=float, default=1)
        parser.add_argument("--receiver-entropy-coeff", type=float, default=0.5)
        parser.add_argument("--receiver-layer-norm", default=False, action="store_true")
        parser.add_argument("--receiver-output-attention", default=False, action="store_true")
        parser.add_argument("--receiver-object-attention", default=False, action="store_true")

        parser.add_argument("--reset-parameters", default=False, action="store_true")
        parser.add_argument("--update-masks", default=False, action="store_true")
        parser.add_argument("--reset-parameters-interval", type=int, default=1,
                            help="Reset interval (in number of epochs)")
        parser.add_argument("--reset-parameters-fraction", type=float, default=0.9)

        return parent_parser

    def init_agents(self):
        if self.params.symmetric:
            raise NotImplementedError()
        else:
            if self.params.optimal_sender:
                raise NotImplementedError()
            else:
                self.senders = ModuleList(
                    [
                        Sender(
                               self.input_size,
                               self.params.vocab_size, self.params.sender_embed_dim,
                               self.params.sender_hidden_dim, self.params.max_len,
                               self.params.sender_layer_norm, self.params.sender_num_layers,
                               self.params.feedback, self.params.vocab_size_feedback)
                        for _ in range(self.params.num_senders)
                    ]
                )
            self.receivers = ModuleList(
                [
                    Receiver(self.params.vocab_size, self.params.receiver_embed_dim,
                             self.params.receiver_hidden_dim, self.params.max_len,
                             self.input_size,
                             self.params.receiver_layer_norm, self.params.receiver_num_layers,
                             self.params.feedback, self.params.vocab_size_feedback,
                             self.params.receiver_object_attention,
                             self.params.receiver_output_attention, self.params.discrimination_num_objects)
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
            optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=self.params.initial_lr) for sender in self.senders]
        optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=self.params.initial_lr) for receiver in self.receivers]

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

    def forward(self, batch, sender_idx, receiver_idx, return_messages=False, disable_noise=False):
        sender_input, receiver_input, labels = batch

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

        if self.params.receiver_starts:
            receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward_first_turn(
                receiver_input)

            if self.params.feedback:
                input_feedback = receiver_output_tokens.detach()
            else:
                input_feedback = None

            sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_first_turn(
                sender_input, input_feedback)

            sender_output_tokens_detached = sender_output_tokens.detach().clone()
            sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)

        else:
            sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_first_turn(sender_input)

            sender_output_tokens_detached = sender_output_tokens.detach().clone()
            sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)

            receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward_first_turn(receiver_input,
                sender_output_tokens_detached)

            if self.params.feedback:
                input_feedback = receiver_output_tokens.detach()
            else:
                input_feedback = None

        receiver_hidden_states[:, 0] = receiver_prev_hidden[-1]

        sender_entropies.append(sender_step_entropy)
        sender_logits.append(sender_step_logits)
        messages_sender.append(sender_output_tokens_detached)
        original_messages_sender.append(sender_output_tokens)
        if self.params.feedback:
            receiver_entropies.append(receiver_step_entropy)
            receiver_logits.append(receiver_step_logits)
            messages_receiver.append(receiver_output_tokens)

        for step in range(1, self.params.max_len):
            if self.params.receiver_starts:
                receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward_subsequent_turn(
                    sender_output_tokens_detached, receiver_prev_hidden, receiver_output_tokens)

                # Do sender forward pass only if we're not in the last iteration (as we want to finish with a receiver
                # forward pass).
                if not step == self.params.max_len - 1:
                    if self.params.feedback:
                        input_feedback = receiver_output_tokens.detach()
                        receiver_entropies.append(receiver_step_entropy)
                        receiver_logits.append(receiver_step_logits)
                        messages_receiver.append(receiver_output_tokens)

                    sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_subsequent_turn(sender_output_tokens, sender_prev_hidden, input_feedback)

                    sender_output_tokens_detached = sender_output_tokens.detach().clone()
                    sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)
                    messages_sender.append(sender_output_tokens_detached)

                    sender_entropies.append(sender_step_entropy)
                    sender_logits.append(sender_step_logits)
                    original_messages_sender.append(sender_output_tokens)
            else:
                sender_output_tokens, sender_step_entropy, sender_step_logits, sender_prev_hidden = sender.forward_subsequent_turn(sender_output_tokens, sender_prev_hidden, input_feedback)

                sender_output_tokens_detached = sender_output_tokens.detach().clone()
                sender_output_tokens_detached = self.add_noise(sender_output_tokens_detached, disable_noise)
                messages_sender.append(sender_output_tokens_detached)

                sender_entropies.append(sender_step_entropy)
                sender_logits.append(sender_step_logits)
                original_messages_sender.append(sender_output_tokens)

                receiver_output_tokens, receiver_step_entropy, receiver_step_logits, receiver_prev_hidden = receiver.forward_subsequent_turn(
                    sender_output_tokens_detached, receiver_prev_hidden, receiver_output_tokens)

                if not step == self.params.max_len - 1:
                    if self.params.feedback:
                        input_feedback = receiver_output_tokens.detach()
                        receiver_entropies.append(receiver_step_entropy)
                        receiver_logits.append(receiver_step_logits)
                        messages_receiver.append(receiver_output_tokens)

            receiver_hidden_states[:, step] = receiver_prev_hidden[-1]

        messages_sender = torch.stack(messages_sender).permute(1, 0)
        sender_logits = torch.stack(sender_logits).permute(1, 0)
        sender_entropies = torch.stack(sender_entropies).permute(1, 0)

        messages_sender_lengths = find_lengths(messages_sender)

        if self.params.feedback:
            messages_receiver = torch.stack(messages_receiver).permute(1, 0)
            receiver_logits = torch.stack(receiver_logits).permute(1, 0)
            receiver_entropies = torch.stack(receiver_entropies).permute(1, 0)

            messages_receiver_lengths = find_lengths(messages_receiver, stop_at_eos=False)
            self.log(f"message_lengths_receiver", messages_receiver_lengths.float().mean())
            self.log(f"message_lengths_sender", messages_sender_lengths.float().mean())

            self.log(f"message_lengths", messages_sender_lengths.float().mean() + messages_receiver_lengths.float().mean())
        else:
            self.log(f"message_lengths", messages_sender_lengths.float().mean())

        for i in range(messages_sender.size(1)):
            sender_entropies[i >= messages_sender_lengths, i] = 0
            sender_logits[i >= messages_sender_lengths, i] = 0

        effective_entropy_s_1 = sender_entropies.sum(dim=1) / messages_sender_lengths.float()
        effective_log_prob_s = sender_logits.sum(dim=1)

        sender_entropy_loss = (effective_entropy_s_1 * self.sender_entropy_coeff).mean()

        policy_length_loss = (messages_sender_lengths.float() * self.length_cost * effective_log_prob_s).mean()

        reward_baseline = self.baselines["reward"].predict(sender_input.device)

        last_time_steps = messages_sender_lengths.numpy().copy()
        if self.params.receiver_starts:
            last_time_steps += 1
        receiver_output = receiver.output(receiver_input, receiver_hidden_states, last_time_steps)
        rewards = (receiver_output.argmax(dim=1) == labels).detach().float()

        receiver_output_loss = F.cross_entropy(receiver_output, labels)

        receiver_loss = receiver_output_loss

        if self.params.feedback:
            effective_entropy_r = receiver_entropies.sum(dim=1) / messages_receiver_lengths.float()
            receiver_entropy_loss = (effective_entropy_r * self.receiver_entropy_coeff).mean()

            receiver_logits = receiver_logits.sum(dim=1)
            receiver_policy_loss = (
                    - rewards * receiver_logits
            ).mean()
            receiver_loss += receiver_policy_loss

            receiver_loss -= receiver_entropy_loss

        self.log(f"receiver_loss", receiver_loss)

        self.log(f"entropy_loss", sender_entropy_loss)

        sender_policy_loss = (- rewards * effective_log_prob_s).mean()

        if self.params.baseline_type == "none":
            baseline_loss = 0
        else:
            baseline_loss = torch.square(rewards - reward_baseline).mean()

        self.log(f"sender_policy_loss", sender_policy_loss)
        self.log(f"sender_policy_length_loss", policy_length_loss)

        sender_loss = sender_policy_loss + baseline_loss + policy_length_loss - sender_entropy_loss

        loss = sender_loss + receiver_loss

        if self.training:
            self.baselines["reward"].update(rewards)

        if return_messages:
            return loss, rewards, messages_sender
        else:
            return loss, rewards

    def add_noise(self, messages, disable_noise=False):
        if not disable_noise and self.params.noise > 0:
            indices = torch.multinomial(torch.tensor([1 - self.params.noise, self.params.noise]),
                                        messages.numel(), replacement=True)
            indices = indices.reshape(messages.shape).to(messages.device)
            # Replace all randomly selected values (but only if they are not EOS symbols (0))
            messages[(indices == 1) & (messages != 0)] = self.token_noise
        return messages

    def update_reset_masks(self):
        self.senders_mask_dict = {}
        self.receivers_mask_dict = {}
        frac = self.hparams.reset_parameters_fraction
        for mask_dict, module in zip([self.senders_mask_dict, self.receivers_mask_dict],
                                     [self.senders, self.receivers]):
            for name, param in module.named_parameters():
                weight_mag = torch.abs(param.detach().clone())
                topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement() * (frac)), largest=False)
                temp_mask = torch.ones(weight_mag.nelement())
                temp_mask[topk.indices] = 0
                mask_dict[name] = temp_mask.bool().view(weight_mag.shape)

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.hparams.reset_parameters:
            self.update_reset_masks()

        if self.hparams.reset_parameters and (self.current_epoch + 1) % self.hparams.reset_parameters_interval == 0:
            if self.hparams.update_masks:
                self.update_reset_masks()

            for mask_dict, module in zip([self.senders_mask_dict, self.receivers_mask_dict], [self.senders, self.receivers]):
                weight_dict = {}
                for name, param in module.named_parameters():
                    weight_dict[name] = param.detach().clone()

                for agent in module:
                    agent.reset_parameters()

                for name, param in module.named_parameters():
                    param.data[mask_dict[name]] = weight_dict[name][mask_dict[name]]

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
            if self.params.noise > 0:
                _, acc_no_noise = self.forward(sender_input, sender_idx, receiver_idx, disable_noise=True)
            else:
                acc_no_noise = acc

            return acc, acc_no_noise

        elif dataloader_idx == 1:
            # Language analysis (on train set data)
            # TODO: do only if required for logging
            _, acc_no_noise, messages = self.forward(sender_input, sender_idx, receiver_idx, return_messages=True, disable_noise=True)

            return sender_input, messages, acc_no_noise

        elif dataloader_idx == 2:
            # Test Generalization
            _, acc = self.forward(sender_input, sender_idx, receiver_idx)
            if self.params.noise > 0:
                _, acc_no_noise = self.forward(sender_input, sender_idx, receiver_idx, disable_noise=True)
            else:
                acc_no_noise = acc

            return acc, acc_no_noise

    def validation_epoch_end(self, validation_step_outputs):
        # Val Generalization:
        accs = torch.cat([acc for acc, _ in validation_step_outputs[0]])
        accs_no_noise = torch.cat([acc_no_noise for _, acc_no_noise in validation_step_outputs[0]])
        val_acc = accs.float().mean().item()
        val_acc_no_noise = accs_no_noise.float().mean().item()
        self.log("val_acc", val_acc, prog_bar=True, add_dataloader_idx=False)
        self.log("val_acc_no_noise", val_acc_no_noise, prog_bar=True, add_dataloader_idx=False)
        is_best_checkpoint = False
        if self.best_val_acc_no_noise < val_acc_no_noise:
            self.best_val_acc_no_noise = val_acc_no_noise
            is_best_checkpoint = True
        self.log("best_val_acc_no_noise", self.best_val_acc_no_noise, prog_bar=True, add_dataloader_idx=False)

        # Language analysis (on train set data)
        language_analysis_results = validation_step_outputs[1]
        train_acc_no_noise = torch.cat([acc for _, _, acc in language_analysis_results])
        self.log("train_acc_no_noise", train_acc_no_noise.float(), prog_bar=True, add_dataloader_idx=False)
        if is_best_checkpoint:
            self.log("train_acc_no_noise_at_best_val_acc", train_acc_no_noise)

        meanings = torch.cat([meaning.cpu() for (meaning, _, _), _, _ in language_analysis_results])
        messages = torch.cat([message.cpu() for _, message, _ in language_analysis_results])
        self.analyze_language(messages, meanings, is_best_checkpoint)

        if len(validation_step_outputs) > 2:
            # Test Generalization:
            accs = torch.cat([acc for acc, _ in validation_step_outputs[2]])
            accs_no_noise = torch.cat([acc_no_noise for _, acc_no_noise in validation_step_outputs[2]])
            test_acc = accs.float().mean().item()
            test_acc_no_noise = accs_no_noise.float().mean().item()
            self.log("test_acc", test_acc, add_dataloader_idx=False)
            self.log("test_acc_no_noise", test_acc_no_noise, add_dataloader_idx=False)

    def analyze_language(self, messages, meanings, is_best_checkpoint=False):
        num_unique_messages = len(messages.unique(dim=0))
        self.log("num_unique_messages", float(num_unique_messages))

        # TODO msgs depend on feedback!
        unique_meanings, indices = torch.unique(meanings, dim=0, return_inverse=True)
        unique_meanings = unique_meanings[:MAX_SAMPLES_LANGUAGE_ANALYSIS]
        unique_messages = []
        for i in range(len(unique_meanings)):
            unique_messages.append(messages[indices == i][0])
        messages = torch.stack(unique_messages)
        meanings = unique_meanings
        # TODO: command line arg:
        # if is_best_checkpoint or self.force_log:
            # meanings_strings = pd.DataFrame(meanings).apply(lambda row: "".join(row.astype(int).astype(str)), axis=1)
            # num_digits = int(math.log10(self.params.vocab_size))
            # messages_strings = pd.DataFrame(messages).apply(lambda row: "".join([s.zfill(num_digits) for s in row.astype(int).astype(str)]), axis=1)
            # messages_df = pd.DataFrame([meanings_strings, messages_strings]).T
            # messages_df.rename(columns={0: 'meaning', 1: 'message'}, inplace=True)
            # messages_df.to_csv(f"{self.logger.log_dir}/messages.csv", index=False)

        if self.params.log_entropy_on_validation or self.force_log:
            entropy = compute_entropy(messages.numpy())
            self.log("message_entropy", entropy, prog_bar=True)
            print("message_entropy: ", entropy)
            if is_best_checkpoint:
                self.log("message_entropy_at_best_val_acc", entropy)

        if self.guesswhat or self.imagenet:
            print("Skipping compositionality metrics calculation.")
        else:
            if self.params.log_topsim_on_validation or self.force_log:
                topsim = compute_topsim(meanings, messages)
                self.log("topsim", topsim, prog_bar=True)
                print("Topsim: ", topsim)
                if is_best_checkpoint:
                    self.log("topsim_at_best_val_acc", topsim)

            if self.params.log_posdis_on_validation or self.force_log:
                posdis = compute_posdis(self.num_attributes, self.num_values, meanings, messages)
                self.log("posdis", posdis, prog_bar=True)
                print("posdis: ", posdis)
                if is_best_checkpoint:
                    self.log("posdis_at_best_val_acc", posdis)

            if self.params.log_bosdis_on_validation or self.force_log:
                bosdis = compute_bosdis(self.num_attributes, self.num_values, meanings, messages, self.params["vocab_size"])
                self.log("bosdis", bosdis, prog_bar=True)
                print("bodis: ", bosdis)
                if is_best_checkpoint:
                    self.log("bosdis_at_best_val_acc", bosdis)

    def on_fit_start(self):
        # Set which metrics to use for hyperparameter tuning
        metrics = ["val_acc_no_noise", "val_acc", "topsim_at_best_val_acc", "posdis_at_best_val_acc",
                   "bosdis_at_best_val_acc", "test_acc_no_noise"]
        self.logger.log_hyperparams(self.hparams, {m: 0 for m in metrics})
