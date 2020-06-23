from collections import OrderedDict
from typing import Union
from pathlib import Path
from math import sqrt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths, dropout_frame
from text.symbols import ctc_symbols


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


# Adapted from
# https://github.com/as-ideas/ForwardTacotron/blob/master/models/forward_tacotron.py
# added positional index
class LengthRegulatorOld(nn.Module):
    def __init__(self, posidx=False):
        super().__init__()
        self.posidx = posidx
        self.posidx_dim = 1 if posidx else 0

    def forward(self, x, dur):
        return self.expand(x, dur)

    def expand(self, x, duration):
        duration[duration < 0] = 0
        tot_duration = duration.cumsum(1).detach().cpu().numpy().astype(
            'int')  # cumulative duration per sample
        # max duration for whole batch batch
        max_duration = int(tot_duration.max().item())
        # batch, timeline, latent dimensions (input text)
        expanded = torch.zeros(x.size(0), max_duration,
                               x.size(2) + self.posidx_dim).to(x.device)

        # loop i - batch, j - time
        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                # cumulative duration of given linguistic element j in batch i
                pos1 = tot_duration[i, j]
                expanded[i, pos:pos1, :expanded.shape[2] -
                         self.posidx_dim] = x[i, j, :].repeat(pos1-pos, 1)
                if self.posidx:
                    expanded[i, pos:pos1, -self.posidx_dim] = torch.linspace(
                        0.0, 1.0, pos1 - pos).to(x.device)
                pos = pos1
        return expanded.to(duration.device)


class LengthRegulator(nn.Module):

    def __init__(self, posidx):
        super().__init__()

    def forward(self, x, dur):
        return self.expand(x, dur)

    @staticmethod
    def build_index(duration, x):
        duration[duration < 0] = 0
        tot_duration = duration.cumsum(1).detach().cpu().numpy().astype('int')
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return torch.LongTensor(index).to(duration.device)

    def expand(self, x, dur):
        idx = self.build_index(dur, x)
        y = torch.gather(x, 1, idx)
        return y


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, in_dims, conv_dims=256, kernel_size=3, dropout=0.1):
        super(DurationPredictor, self).__init__()

        self.input_size = in_dims
        self.filter_size = conv_dims
        self.kernel = kernel_size
        self.conv_output_size = conv_dims
        self.dropout = dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output, alpha=1.0):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out / alpha


class DurationPredictorRNN(nn.Module):
    def __init__(self, in_dims, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims,
                          batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int(
                                 (hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = nn.Sequential(
            LinearNorm(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
                       hparams.decoder_rnn_dim, bias=True, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.mel_layer = nn.Sequential(
            LinearNorm(hparams.decoder_rnn_dim,
                       hparams.decoder_rnn_dim, bias=True, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            LinearNorm(hparams.decoder_rnn_dim,
                       hparams.n_mel_channels * hparams.n_frames_per_step)
        )

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim, 1, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, decoder_outputs, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        decoder_outputs = torch.stack(
            decoder_outputs).transpose(0, 1).contiguous()
        decoder_outputs = decoder_outputs.transpose(1, 2)

        return decoder_outputs, mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        mel_output = self.mel_layer(decoder_output)
        gate_prediction = self.gate_layer(decoder_output)
        return decoder_output, mel_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        decoder_outputs, mel_outputs, gate_outputs, alignments = [], [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (decoder_output, mel_output, gate_output,
             attention_weights) = self.decode(decoder_input)
            decoder_outputs += [decoder_output.squeeze(1)]
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        (decoder_outputs, mel_outputs, gate_outputs, alignments
         ) = self.parse_decoder_outputs(decoder_outputs, mel_outputs, gate_outputs, alignments)

        return decoder_outputs, mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        decoder_outputs, mel_outputs, gate_outputs, alignments = [], [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            decoder_output, mel_output, gate_output, alignment = self.decode(
                decoder_input)

            decoder_outputs += [decoder_output.squeeze(1)]
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        (decoder_outputs, mel_outputs, gate_outputs, alignments
         ) = self.parse_decoder_outputs(decoder_outputs, mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class MIEsitmator(nn.Module):
    def __init__(self, vocab_size, decoder_dim, hidden_size, dropout=0.5):
        super(MIEsitmator, self).__init__()
        self.proj = nn.Sequential(
            LinearNorm(decoder_dim, hidden_size,
                       bias=True, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.ctc_proj = LinearNorm(hidden_size, vocab_size + 1, bias=True)
        self.ctc = nn.CTCLoss(blank=vocab_size, reduction='none')

    def forward(self, decoder_outputs, target_phones, decoder_lengths, target_lengths):
        out = self.proj(decoder_outputs)
        log_probs = self.ctc_proj(out).log_softmax(dim=2)
        log_probs = log_probs.transpose(1, 0)
        ctc_loss = self.ctc(log_probs, target_phones,
                            decoder_lengths, target_lengths)
        # average by number of frames since taco_loss is averaged.
        ctc_loss = (ctc_loss / decoder_lengths.float()).mean()
        return ctc_loss


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.drop_frame_rate = hparams.drop_frame_rate
        self.use_mmi = hparams.use_mmi
        self.hparams = hparams
        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        if self.use_mmi:
            vocab_size = len(ctc_symbols)
            decoder_dim = hparams.decoder_rnn_dim
            self.mi = MIEsitmator(vocab_size, decoder_dim,
                                  decoder_dim, dropout=0.5)
        else:
            self.mi = None

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text, ctc_text_lengths, ids, durs = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        ctc_text = to_gpu(ctc_text).long()
        ctc_text_lengths = to_gpu(ctc_text_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             ctc_text, ctc_text_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mel_mask = mask.expand(self.n_mel_channels,
                                   mask.size(0), mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            if outputs[0] is not None:
                float_mask = (~mask).float().unsqueeze(1)
                outputs[0] = outputs[0] * float_mask
            outputs[1].data.masked_fill_(mel_mask, 0.0)
            outputs[2].data.masked_fill_(mel_mask, 0.0)
            outputs[3].data.masked_fill_(
                mel_mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, *_ = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        if self.drop_frame_rate > 0. and self.training:
            # mels shape (B, n_mel_channels, T_out),
            if self.global_mean is None:
                self.global_mean = getattr(self.hparams, 'global_mean', None)
            mels = dropout_frame(mels, self.global_mean,
                                 output_lengths, self.drop_frame_rate)

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        (decoder_outputs, mel_outputs, gate_outputs, alignments
         ) = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [decoder_outputs, mel_outputs,
                mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        # keep the original interface
        return outputs[1:]


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(
            len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(
            proj_channels[0], proj_channels[1], 3, relu=False)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(
                proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels,
                          batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x)  # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]


class ForwardTacotron(nn.Module):
    def __init__(self,
                 hparams,
                 num_chars,
                 embed_dims=256,
                 durpred_conv_dims=256,
                 durpred_rnn_dims=64,
                 durpred_dropout=0.1,
                 rnn_dim=512,
                 prenet_k=16,
                 prenet_dims=256,
                 postnet_k=8,
                 postnet_dims=256,
                 highways=4,
                 dropout=0.1,
                 n_mels=80):

        super().__init__()
        self.rnn_dim = rnn_dim
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator(hparams.positional_index)
        self.positional_index_dim = 1 if hparams.positional_index else 0
        # self.dur_pred = DurationPredictor(2 * prenet_dims,
        #                                  conv_dims=durpred_conv_dims,
        #                                  rnn_dims=durpred_rnn_dims,
        #                                  dropout=durpred_dropout)
        self.dur_pred = DurationPredictor(
            2 * prenet_dims, dropout=durpred_dropout)
        self.prenet = Prenet(embed_dims,
                             [hparams.prenet_dim, hparams.prenet_dim])
        self.cbhg = CBHG(K=prenet_k,
                         in_channels=hparams.prenet_dim,
                         channels=prenet_dims,
                         proj_channels=[prenet_dims, embed_dims],
                         num_highways=highways)
        self.lstm = nn.LSTM(2 * prenet_dims + self.positional_index_dim,
                            rnn_dim,
                            batch_first=True,
                            bidirectional=True)
        self.lin = torch.nn.Linear(2 * rnn_dim, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=highways)
        self.dropout = dropout
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)
        self.mi = None

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, ctc_text, ctc_text_lengths, durs = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        if self.training:
            self.step += 1

        x = self.embedding(text_inputs)

        x = self.prenet(x)
        x = x.transpose(1, 2)
        x = self.cbhg(x)

        # since 2020/06/19 we feed prenet inputs into dur pred
        # as described in https://github.com/as-ideas/ForwardTacotron/issues/11
        dur_hat = self.dur_pred(x)
        dur_hat = dur_hat.squeeze()

        x = self.lr(x, durs)
        x, _ = self.lstm(x)
        #x = F.dropout(x,
        #              p=self.dropout,
        #              training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self.pad(x_post, mels.size(2))
        x = self.pad(x, mels.size(2))
        #x_post = x
        return (x, x_post, dur_hat)

    def inference(self, x, alpha=1.0):
        self.eval()
        # use same device as parameters
        #device = next(self.parameters()).device
        #x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        print(f"x init {x.shape}")
        x = self.embedding(x)  # .transpose(1,2)

        print(f"x pre prenet {x.shape}")
        x = self.prenet(x)
        x = x.transpose(1, 2)
        x = self.cbhg(x)

        print(f"x pre dur pred {x.shape}")
        dur = self.dur_pred(x, alpha=alpha)
        #dur = dur.squeeze(2)
        #dur = dur.squeeze()

        print(f"x pre lr {x.shape}")
        x = self.lr(x, dur)
        print(f"x pre lstm {x.shape}")
        x, _ = self.lstm(x)
        #x = F.dropout(x,
        #              p=self.dropout,
        #              training=self.training)
        print(f"x pre lin {x.shape}")
        x = self.lin(x)
        x = x.transpose(1, 2)

        print(f"x pre postnet {x.shape}")
        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)
        #x_post = x

        print(f"x post {x_post.shape}")
        #x, x_post, dur = x.squeeze(), x_post.squeeze(), dur.squeeze()
        #x = x.cpu().data.numpy()
        #x_post = x_post.cpu().data.numpy()
        #dur = dur.cpu().data.numpy()

        return x, x_post, None, dur

    def pad(self, x, max_len):
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', 0.0)
        return x

    def get_step(self):
        return self.step.data.item()

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text, ctc_text_lengths, ids, durs = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        durs_padded = to_gpu(durs).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        ctc_text = to_gpu(ctc_text).long()
        ctc_text_lengths = to_gpu(ctc_text_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             ctc_text, ctc_text_lengths, durs_padded),
            (mel_padded, gate_padded))


class DurationTacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        #self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(embed_dims,
                                          conv_dims=durpred_conv_dims,
                                          rnn_dims=durpred_rnn_dims,
                                          dropout=durpred_dropout)
        self.lstm = nn.LSTM(2 * prenet_dims,
                            rnn_dim,
                            batch_first=True,
                            bidirectional=True)
        self.drop_frame_rate = hparams.drop_frame_rate
        self.use_mmi = hparams.use_mmi
        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        if self.use_mmi:
            vocab_size = len(ctc_symbols)
            decoder_dim = hparams.decoder_rnn_dim
            self.mi = MIEsitmator(vocab_size, decoder_dim,
                                  decoder_dim, dropout=0.5)
        else:
            self.mi = None

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text, ctc_text_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        ctc_text = to_gpu(ctc_text).long()
        ctc_text_lengths = to_gpu(ctc_text_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             ctc_text, ctc_text_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mel_mask = mask.expand(self.n_mel_channels,
                                   mask.size(0), mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            if outputs[0] is not None:
                float_mask = (~mask).float().unsqueeze(1)
                outputs[0] = outputs[0] * float_mask
            outputs[1].data.masked_fill_(mel_mask, 0.0)
            outputs[2].data.masked_fill_(mel_mask, 0.0)
            outputs[3].data.masked_fill_(
                mel_mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, durations, *_ = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        # dur predictions
        dur_hat = self.dur_pred(text_inputs)
        dur_hat = dur_hat.squeeze()

        print(f"text input shape {text_inputs.shape}")
        # embedding -> encoder (conv + blstm) -> duration expansion
        # -> 2 LSTM layers (originally decoder) -> linear projection -> postnet with skip
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        print(f"embedded input shape {embedded_inputs.shape}")

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        print(f"encoder outputs shape {encoder_outputs.shape}")

        # prenet?
        # x = self.prenet(x)

        # instead of attention we expand states
        expanded_encoder_outputs = self.lr(encoder_outputs, durations)

        # "decoder" blstms and lin project
        decoder_outputs, _ = self.lstm(expanded_encoder_outputs)
        decoder_outputs = F.dropout(decoder_outputs,
                                    p=self.dropout,
                                    training=self.training)
        mel_outputs = self.lin(decoder_outputs)
        #x = x.transpose(1, 2)

        # postnet with residual
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        #x_post = self.pad(x_post, mel.size(2))
        #x = self.pad(x, mel.size(2))
        return mel_outputs, mel_outputs_postnet, dur_hat

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        # keep the original interface
        return outputs[1:]
